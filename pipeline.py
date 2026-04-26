#!/usr/bin/env python3
"""
pipeline.py -- Main orchestrator for automated research loop.

Inspired by AutoResearchClaw's multi-stage pipeline, adapted for
the autoresearch-katze attack optimization framework.

Usage:
    python pipeline.py --max-iterations 20 --target-asr 0.95
    python pipeline.py --step                    # Run one iteration
    python pipeline.py --analyze-only            # Build knowledge, no experiments
    python pipeline.py --dry-run                 # Show next hypothesis
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from knowledge import build_knowledge_base, save_knowledge, load_knowledge
from hypothesis import HypothesisGenerator, parse_program_ideas
from codegen import (
    backup_attack, apply_config_changes, apply_code_patch,
    validate_attack_py, restore_attack, read_current_config,
)
from executor import run_experiment
from analyzer import ResultAnalyzer
from decision import DecisionEngine, Action

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "pipeline.log"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def print_knowledge_summary(kb):
    """Print human-readable knowledge summary."""
    print(f"\n{'=' * 60}")
    print(f"  KNOWLEDGE BASE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total experiments: {kb.total_experiments}")
    print(f"  Best 5-fold ASR:  {kb.best_5fold_asr:.2%}")
    print(f"  Best 1-fold ASR:  {kb.best_1fold_asr:.2%}")
    print(f"  Quick→Full bias:  {kb.quick_to_full_bias:.4f}")
    print(f"  No-improvement streak: {kb.consecutive_no_improvement}")
    print(f"  Lessons extracted: {len(kb.lessons)}")

    print(f"\n  Explored parameter ranges:")
    for p, vals in sorted(kb.param_ranges_explored.items()):
        print(f"    {p}: {vals}")

    print(f"\n  Unexplored values:")
    for p, vals in sorted(kb.unexplored_param_values.items()):
        print(f"    {p}: {vals}")

    if kb.best_5fold_config:
        print(f"\n  Best 5-fold config:")
        for k, v in sorted(kb.best_5fold_config.items()):
            print(f"    {k}: {v}")

    print(f"\n  Key lessons:")
    # Show param_effect lessons with highest absolute delta
    effects = [l for l in kb.lessons if l.category == "param_effect" and l.asr_delta]
    effects.sort(key=lambda l: abs(l.asr_delta or 0), reverse=True)
    for l in effects[:10]:
        conf = f"[{l.confidence}]"
        print(f"    {conf:6s} {l.description}")

    trends = [l for l in kb.lessons if l.category == "trend"]
    if trends:
        print(f"\n  Trends:")
        for l in trends:
            print(f"    {l.description}")

    print(f"{'=' * 60}\n")


def run_pipeline(max_iterations: int = 20,
                 target_asr: float = 0.95,
                 step: bool = False,
                 dry_run: bool = False,
                 analyze_only: bool = False,
                 verbose: bool = False):
    """Main pipeline loop."""
    setup_logging(verbose)
    log = logging.getLogger("pipeline")

    # ── Phase 1: Build knowledge base ──
    log.info("Building knowledge base...")
    kb = build_knowledge_base()
    save_knowledge(kb)
    log.info("Knowledge: %d experiments, best_5fold=%.4f, %d lessons",
             kb.total_experiments, kb.best_5fold_asr, len(kb.lessons))

    if analyze_only:
        print_knowledge_summary(kb)
        return

    # ── Phase 2: Initialize components ──
    program_ideas = parse_program_ideas(str(BASE_DIR / "program.md"))
    hyp_gen = HypothesisGenerator(kb, program_ideas)
    analyzer = ResultAnalyzer(kb)
    decision_engine = DecisionEngine(target_asr=target_asr)

    consecutive_failures = 0
    no_improvement_count = kb.consecutive_no_improvement
    pending_promote = False
    last_result = None

    if dry_run:
        hypothesis = hyp_gen.select_next()
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — Next hypothesis:")
        print(f"{'=' * 60}")
        print(f"  Name:      {hypothesis.name}")
        print(f"  Strategy:  {hypothesis.strategy}")
        print(f"  Rationale: {hypothesis.rationale}")
        print(f"  Changes:   {json.dumps(hypothesis.config_changes, indent=4)}")
        print(f"  Run mode:  {hypothesis.run_mode}")
        print(f"  Priority:  {hypothesis.priority}")

        # Show all candidates
        candidates = hyp_gen.generate_candidates()
        print(f"\n  All candidates ({len(candidates)}):")
        for i, c in enumerate(candidates[:15]):
            marker = ">>>" if c.name == hypothesis.name else "   "
            print(f"  {marker} [{c.priority}] {c.strategy:12s} {c.name}")
        print(f"{'=' * 60}\n")
        return

    # ── Phase 3: Main loop ──
    for iteration in range(1, max_iterations + 1):
        log.info("=" * 60)
        log.info("ITERATION %d / %d", iteration, max_iterations)
        log.info("=" * 60)

        # Generate or promote
        if pending_promote:
            log.info("PROMOTING to full 5-fold evaluation")
            hypothesis = None
            run_mode = "full"
            pending_promote = False
        else:
            hypothesis = hyp_gen.select_next()
            run_mode = hypothesis.run_mode
            log.info("Hypothesis: %s [%s]", hypothesis.name, hypothesis.strategy)
            log.info("  Rationale: %s", hypothesis.rationale)
            if hypothesis.config_changes:
                log.info("  Changes: %s", hypothesis.config_changes)

        # Apply changes with backup
        backup = backup_attack()
        if hypothesis and hypothesis.config_changes:
            apply_config_changes(hypothesis.config_changes)
            log.info("Applied config changes to attack.py")

        if hypothesis and hypothesis.code_patch_old and hypothesis.code_patch_new:
            ok = apply_code_patch(hypothesis.code_patch_old, hypothesis.code_patch_new)
            if not ok:
                log.warning("Code patch failed to apply, reverting")
                restore_attack(backup)
                consecutive_failures += 1
                continue

        # Validate syntax
        if not validate_attack_py():
            log.error("attack.py syntax invalid, reverting")
            restore_attack(backup)
            consecutive_failures += 1
            continue

        # Execute
        is_quick = (run_mode == "quick")
        log.info("Running experiment (%s)...", "quick" if is_quick else "full 5-fold")
        exec_result = run_experiment(
            quick=is_quick, timeout_s=600, backup_path=backup)

        if not exec_result.success:
            log.error("FAILED: %s — %s", exec_result.error_type, exec_result.error_message)
            if exec_result.reverted:
                log.info("Auto-reverted attack.py from backup")
            consecutive_failures += 1
            no_improvement_count += 1
            continue

        consecutive_failures = 0

        # Analyze
        report = analyzer.analyze(exec_result.result_data, hypothesis)
        log.info("Result: ASR=%.4f (std=%.4f) [%s] wall=%.1fs",
                 report.mean_asr, report.std_asr,
                 "quick" if report.is_quick_run else f"{report.n_folds}-fold",
                 report.wall_time)
        if report.verification:
            log.info("Verification: %s (integrity=%.0f%%)",
                     report.verification.overall_status.value,
                     report.verification.integrity_score * 100)
        if report.suspicious:
            log.warning("SUSPICIOUS: %s", report.suspicious_reason)
        if report.fold_variance_flag:
            log.warning("High fold variance: std=%.4f", report.std_asr)
        if report.is_new_best:
            log.info("*** NEW BEST 5-fold ASR: %.4f ***", report.mean_asr)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Evolution: extract lessons
        from evolution import EvolutionStore, extract_lessons_from_result
        evo_store = EvolutionStore()
        lessons = extract_lessons_from_result(
            exec_result.result_data,
            prev_result=last_result,
        )
        if lessons:
            evo_store.append_many(lessons)
            log.info("Extracted %d evolution lessons", len(lessons))

        # Update knowledge
        kb = build_knowledge_base()
        save_knowledge(kb)
        analyzer.knowledge = kb
        hyp_gen.knowledge = kb

        # Decide
        decision = decision_engine.decide(
            report, kb, consecutive_failures, no_improvement_count)
        log.info("Decision: %s — %s", decision.action.value, decision.reason)

        if decision.action == Action.STOP:
            log.info("TARGET REACHED. Stopping pipeline.")
            break
        elif decision.action == Action.PROMOTE:
            pending_promote = True
        elif decision.action == Action.PIVOT:
            hyp_gen.force_explore = True
            log.info("Pivoting to exploration mode")

        last_result = exec_result.result_data

        if step:
            log.info("Step mode: stopping after 1 iteration")
            break

    # ── Final: generate charts, review, deliverables ──
    kb = build_knowledge_base()
    save_knowledge(kb)
    log.info("Generating final artifacts...")

    try:
        from charts import generate_all_charts
        chart_paths = generate_all_charts()
        log.info("Generated %d charts", len(chart_paths))
    except Exception as e:
        log.warning("Chart generation failed: %s", e)

    try:
        from review import generate_review
        review_path = generate_review()
        log.info("Generated review: %s", review_path)
    except Exception as e:
        log.warning("Review generation failed: %s", e)

    try:
        from deliverables import package_deliverables
        pkg_path = package_deliverables()
        log.info("Deliverables packaged: %s", pkg_path)
    except Exception as e:
        log.warning("Deliverables packaging failed: %s", e)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Iterations run:   {iteration}")
    print(f"  Best 5-fold ASR:  {kb.best_5fold_asr:.2%}")
    print(f"  Target ASR:       {target_asr:.2%}")
    print(f"  Target reached:   {'YES' if kb.best_5fold_asr >= target_asr else 'NO'}")
    print(f"  Total experiments: {kb.total_experiments}")
    print(f"  Deliverables:     deliverables/")
    print(f"  Charts:           charts/")
    print(f"  Review:           reviews.md")
    print(f"  Evolution:        evolution/lessons.jsonl")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoResearch-Katze Pipeline — Automated attack optimization")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Maximum pipeline iterations (default: 20)")
    parser.add_argument("--target-asr", type=float, default=0.95,
                        help="Target 5-fold ASR to achieve (default: 0.95)")
    parser.add_argument("--step", action="store_true",
                        help="Run one iteration only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show next hypothesis without running")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only build and display knowledge base")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--charts", action="store_true",
                        help="Generate charts only")
    parser.add_argument("--review", action="store_true",
                        help="Generate peer review only")
    parser.add_argument("--package", action="store_true",
                        help="Package deliverables only")
    args = parser.parse_args()

    # Standalone actions
    if args.charts:
        from charts import generate_all_charts
        generate_all_charts()
        sys.exit(0)
    if args.review:
        from review import generate_review
        path = generate_review()
        print(path.read_text())
        sys.exit(0)
    if args.package:
        from deliverables import package_deliverables
        package_deliverables()
        sys.exit(0)

    run_pipeline(
        max_iterations=args.max_iterations,
        target_asr=args.target_asr,
        step=args.step,
        dry_run=args.dry_run,
        analyze_only=args.analyze_only,
        verbose=args.verbose,
    )
