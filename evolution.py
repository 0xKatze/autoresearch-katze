#!/usr/bin/env python3
"""
evolution.py -- Self-learning lesson extraction from experiments.

Inspired by AutoResearchClaw's evolution.py + MetaClaw bridge.
Extracts lessons from experiment results, stores as JSONL, and generates
prompt overlays for hypothesis generation.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

EVOLUTION_DIR = Path(__file__).parent / "evolution"
LESSONS_FILE = EVOLUTION_DIR / "lessons.jsonl"


class LessonCategory(Enum):
    PARAM_TUNING = "param_tuning"       # Hyperparameter insights
    STRATEGY = "strategy"                # Attack strategy insights
    FAILURE = "failure"                  # What went wrong
    ARCHITECTURE = "architecture"        # Generator/model insights
    STATISTICAL = "statistical"          # Evaluation methodology
    BOTTLENECK = "bottleneck"            # Performance bottlenecks


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Lesson:
    timestamp: str
    category: str          # LessonCategory value
    severity: str          # Severity value
    description: str       # Human-readable lesson
    evidence: str          # What data supports this
    actionable: str        # What to do about it
    source_experiment: str  # Experiment timestamp
    asr_context: float     # ASR at time of lesson


class EvolutionStore:
    """Append-only JSONL lesson store."""

    def __init__(self, path: Path = LESSONS_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, lesson: Lesson):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(lesson)) + "\n")

    def append_many(self, lessons: List[Lesson]):
        with open(self.path, "a") as f:
            for l in lessons:
                f.write(json.dumps(asdict(l)) + "\n")

    def load_all(self) -> List[Lesson]:
        if not self.path.exists():
            return []
        lessons = []
        for line in self.path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                d = json.loads(line)
                lessons.append(Lesson(**d))
            except (json.JSONDecodeError, TypeError):
                continue
        return lessons

    def load_recent(self, n: int = 20) -> List[Lesson]:
        all_lessons = self.load_all()
        return all_lessons[-n:]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text().strip().split("\n") if line)


def extract_lessons_from_result(result: Dict,
                                 prev_result: Optional[Dict] = None,
                                 knowledge: Optional[Dict] = None) -> List[Lesson]:
    """Extract lessons from a single experiment result."""
    lessons = []
    ts = datetime.now().isoformat()
    exp_ts = result.get("timestamp", "unknown")
    asr = result.get("mean_asr", 0)
    std = result.get("std_asr", 0)
    nf = result.get("n_folds", 1)
    cfg = result.get("config", {})
    wall = result.get("wall_time_s", 0)

    # ── Failure patterns ──
    if asr == 0:
        lessons.append(Lesson(
            timestamp=ts, category=LessonCategory.FAILURE.value,
            severity=Severity.CRITICAL.value,
            description="Attack produced zero ASR — completely ineffective",
            evidence=f"Config: {json.dumps(cfg, default=str)}",
            actionable="Revert this config immediately; check if attack logic is broken",
            source_experiment=exp_ts, asr_context=asr,
        ))

    if nf >= 5 and std > 0.08:
        lessons.append(Lesson(
            timestamp=ts, category=LessonCategory.STATISTICAL.value,
            severity=Severity.WARNING.value,
            description=f"High fold variance: std={std:.4f} across {nf} folds",
            evidence=f"Per-fold ASRs: {result.get('per_fold_asr', [])}",
            actionable="Focus on reducing fold variance; some folds may have harder graphs",
            source_experiment=exp_ts, asr_context=asr,
        ))

    if nf == 1 and asr >= 1.0:
        lessons.append(Lesson(
            timestamp=ts, category=LessonCategory.STATISTICAL.value,
            severity=Severity.WARNING.value,
            description="100% ASR on 1-fold — likely overfit to easy fold",
            evidence=f"1-fold ASR={asr:.4f}, must verify with 5-fold",
            actionable="Always confirm with full 5-fold before declaring improvement",
            source_experiment=exp_ts, asr_context=asr,
        ))

    # ── Parameter insights (compare with previous) ──
    if prev_result:
        prev_asr = prev_result.get("mean_asr", 0)
        prev_cfg = prev_result.get("config", {})
        delta = asr - prev_asr

        changed = {k: (prev_cfg.get(k), cfg.get(k))
                   for k in set(list(cfg.keys()) + list(prev_cfg.keys()))
                   if cfg.get(k) != prev_cfg.get(k)}

        if len(changed) == 1:
            param, (old, new) = list(changed.items())[0]
            if delta > 0.02:
                lessons.append(Lesson(
                    timestamp=ts, category=LessonCategory.PARAM_TUNING.value,
                    severity=Severity.INFO.value,
                    description=f"Changing {param}: {old} → {new} improved ASR by {delta:+.4f}",
                    evidence=f"ASR: {prev_asr:.4f} → {asr:.4f}",
                    actionable=f"Continue exploring {param} around {new}",
                    source_experiment=exp_ts, asr_context=asr,
                ))
            elif delta < -0.02:
                lessons.append(Lesson(
                    timestamp=ts, category=LessonCategory.PARAM_TUNING.value,
                    severity=Severity.WARNING.value,
                    description=f"Changing {param}: {old} → {new} HURT ASR by {delta:+.4f}",
                    evidence=f"ASR: {prev_asr:.4f} → {asr:.4f}",
                    actionable=f"Avoid {param}={new}; revert to {old}",
                    source_experiment=exp_ts, asr_context=asr,
                ))

    # ── Strategy patterns ──
    if cfg.get("edge_strategy") == "full" and asr > 0.9:
        lessons.append(Lesson(
            timestamp=ts, category=LessonCategory.STRATEGY.value,
            severity=Severity.INFO.value,
            description="Full edge connectivity effective at high ASR regime",
            evidence=f"ASR={asr:.4f} with edge_strategy=full",
            actionable="Keep full connectivity as baseline strategy",
            source_experiment=exp_ts, asr_context=asr,
        ))

    # ── Bottleneck detection ──
    if nf >= 5 and asr > 0.9 and std > 0.05:
        per_fold = result.get("per_fold_asr", [])
        if per_fold:
            worst_fold = min(per_fold)
            best_fold = max(per_fold)
            if best_fold - worst_fold > 0.15:
                lessons.append(Lesson(
                    timestamp=ts, category=LessonCategory.BOTTLENECK.value,
                    severity=Severity.WARNING.value,
                    description=f"Large fold gap: best={best_fold:.2%} vs worst={worst_fold:.2%}",
                    evidence=f"Per-fold: {[f'{x:.2%}' for x in per_fold]}",
                    actionable="Investigate worst fold — may need fold-specific hyperparams",
                    source_experiment=exp_ts, asr_context=asr,
                ))

    # ── Wall time anomalies ──
    if nf >= 5 and wall > 600:
        lessons.append(Lesson(
            timestamp=ts, category=LessonCategory.FAILURE.value,
            severity=Severity.WARNING.value,
            description=f"Slow experiment: {wall:.0f}s for {nf} folds",
            evidence=f"Config: epochs={cfg.get('attack_epochs')}, grad={cfg.get('grad_method')}",
            actionable="Reduce attack_epochs or switch to faster gradient method",
            source_experiment=exp_ts, asr_context=asr,
        ))

    return lessons


def build_overlay(store: EvolutionStore, max_lessons: int = 8) -> str:
    """Generate a prompt overlay from recent lessons for hypothesis generation."""
    recent = store.load_recent(max_lessons * 2)
    if not recent:
        return ""

    # Prioritize by severity
    severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    recent.sort(key=lambda l: severity_order.get(l.severity, 4))
    selected = recent[:max_lessons]

    lines = ["LESSONS FROM PRIOR EXPERIMENTS:", ""]
    for i, l in enumerate(selected, 1):
        lines.append(f"{i}. [{l.severity}] {l.description}")
        lines.append(f"   → {l.actionable}")
        lines.append("")

    lines.append("MITIGATION: Ensure your next hypothesis addresses these lessons.")
    return "\n".join(lines)


RESULTS_DIR = Path(__file__).parent / "results"


def extract_lessons_from_history(results_dir: Path = RESULTS_DIR) -> List[Lesson]:
    """Batch extract lessons from all experiment results."""
    results = []
    for p in sorted(results_dir.glob("exp_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda r: r.get("timestamp", ""))

    all_lessons = []
    for i, r in enumerate(results):
        prev = results[i - 1] if i > 0 else None
        lessons = extract_lessons_from_result(r, prev)
        all_lessons.extend(lessons)
    return all_lessons

if __name__ == "__main__":
    store = EvolutionStore()
    lessons = extract_lessons_from_history()
    store.append_many(lessons)
    print(f"Extracted {len(lessons)} lessons from experiment history")
    print(f"Total stored: {store.count()}")

    overlay = build_overlay(store)
    if overlay:
        print(f"\nPrompt overlay:\n{overlay}")
