#!/usr/bin/env python3
"""
deliverables.py -- Package all outputs into a deliverables/ folder.

Inspired by AutoResearchClaw's export/publish stage.
Collects: charts, reviews, knowledge, best config, evolution lessons,
and experiment summaries into one compile-ready folder.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).parent
DELIVERABLES_DIR = BASE_DIR / "deliverables"


def _copy_charts(output_dir: Path) -> List[Path]:
    """Copy all charts to deliverables."""
    charts_src = BASE_DIR / "charts"
    charts_dst = output_dir / "charts"
    copied = []
    if charts_src.exists():
        charts_dst.mkdir(parents=True, exist_ok=True)
        for f in charts_src.glob("*.png"):
            dst = charts_dst / f.name
            shutil.copy2(f, dst)
            copied.append(dst)
    return copied


def _copy_reviews(output_dir: Path) -> Path:
    """Copy reviews.md to deliverables."""
    src = BASE_DIR / "reviews.md"
    dst = output_dir / "reviews.md"
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def _generate_experiment_summary(output_dir: Path) -> Path:
    """Generate a structured experiment summary JSON."""
    from knowledge import build_knowledge_base
    kb = build_knowledge_base()

    # Collect full results
    full_results = []
    for p in sorted((BASE_DIR / "results").glob("exp_*.json")):
        try:
            r = json.loads(p.read_text())
            if r.get("n_folds", 1) >= 5:
                full_results.append(r)
        except (json.JSONDecodeError, OSError):
            continue

    full_results.sort(key=lambda r: r.get("mean_asr", 0), reverse=True)

    summary = {
        "generated": datetime.now().isoformat(),
        "dataset": "PROTEINS",
        "victim_model": "GCN (3-layer, 64-dim)",
        "attack_type": "Black-box node injection (ZOO)",
        "evaluation": "5-fold stratified cross-validation",
        "total_experiments": kb.total_experiments,
        "best_result": {
            "mean_asr": kb.best_5fold_asr,
            "config": kb.best_5fold_config,
            "timestamp": kb.best_5fold_timestamp,
        },
        "top_5_results": [
            {
                "mean_asr": r.get("mean_asr"),
                "std_asr": r.get("std_asr"),
                "config": r.get("config"),
                "timestamp": r.get("timestamp"),
            }
            for r in full_results[:5]
        ],
        "key_findings": [l.description for l in kb.lessons[:10]],
        "unexplored_params": kb.unexplored_param_values,
    }

    path = output_dir / "experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def _copy_evolution(output_dir: Path) -> Path:
    """Copy evolution lessons to deliverables."""
    evo_dir = output_dir / "evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)
    src = BASE_DIR / "evolution" / "lessons.jsonl"
    dst = evo_dir / "lessons.jsonl"
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def _copy_knowledge(output_dir: Path) -> Path:
    """Copy knowledge.json to deliverables."""
    src = BASE_DIR / "knowledge.json"
    dst = output_dir / "knowledge.json"
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def _copy_best_attack(output_dir: Path) -> Path:
    """Copy current attack.py as the best attack code."""
    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    src = BASE_DIR / "attack.py"
    dst = code_dir / "attack.py"
    shutil.copy2(src, dst)

    # Also copy best.json
    best_src = BASE_DIR / "best.json"
    if best_src.exists():
        shutil.copy2(best_src, code_dir / "best.json")

    return dst


def _generate_manifest(output_dir: Path, files: List[Path]) -> Path:
    """Generate a manifest of all deliverables."""
    manifest = {
        "generated": datetime.now().isoformat(),
        "project": "autoresearch-katze",
        "description": "Black-box node injection attack on graph classification (PROTEINS/GCN)",
        "files": [],
    }

    for f in sorted(files):
        if f.exists():
            rel = f.relative_to(output_dir)
            manifest["files"].append({
                "path": str(rel),
                "size_bytes": f.stat().st_size,
                "type": f.suffix.lstrip(".") or "directory",
            })

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def package_deliverables(output_dir: Path = DELIVERABLES_DIR,
                          generate_missing: bool = True) -> Path:
    """Package all outputs into deliverables folder.

    Args:
        output_dir: Where to create the package
        generate_missing: If True, generate charts/reviews/evolution if not present

    Returns:
        Path to deliverables directory
    """
    # Clean and recreate
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_files: List[Path] = []

    # Generate missing artifacts
    if generate_missing:
        # Charts
        if not (BASE_DIR / "charts").exists():
            print("Generating charts...")
            from charts import generate_all_charts
            generate_all_charts()

        # Reviews
        if not (BASE_DIR / "reviews.md").exists():
            print("Generating reviews...")
            from review import generate_review
            generate_review()

        # Evolution
        if not (BASE_DIR / "evolution" / "lessons.jsonl").exists():
            print("Extracting evolution lessons...")
            from evolution import EvolutionStore, extract_lessons_from_history
            store = EvolutionStore()
            lessons = extract_lessons_from_history()
            store.append_many(lessons)

        # Knowledge
        if not (BASE_DIR / "knowledge.json").exists():
            print("Building knowledge base...")
            from knowledge import build_knowledge_base, save_knowledge
            kb = build_knowledge_base()
            save_knowledge(kb)

    # Copy artifacts
    print("Packaging deliverables...")

    # Charts
    charts = _copy_charts(output_dir)
    all_files.extend(charts)
    print(f"  Charts: {len(charts)} files")

    # Reviews
    reviews = _copy_reviews(output_dir)
    if reviews.exists():
        all_files.append(reviews)
        print(f"  Reviews: {reviews.name}")

    # Experiment summary
    summary = _generate_experiment_summary(output_dir)
    all_files.append(summary)
    print(f"  Summary: {summary.name}")

    # Evolution lessons
    evo = _copy_evolution(output_dir)
    if evo.exists():
        all_files.append(evo)
        print(f"  Evolution: {evo.name}")

    # Knowledge base
    kb = _copy_knowledge(output_dir)
    if kb.exists():
        all_files.append(kb)
        print(f"  Knowledge: {kb.name}")

    # Best attack code
    code = _copy_best_attack(output_dir)
    all_files.append(code)
    print(f"  Code: {code.name}")

    # Manifest
    manifest = _generate_manifest(output_dir, all_files)
    print(f"  Manifest: {manifest.name}")

    print(f"\nDeliverables packaged: {output_dir}/")
    print(f"Total files: {len(all_files) + 1}")

    return output_dir


if __name__ == "__main__":
    package_deliverables()
