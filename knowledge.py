#!/usr/bin/env python3
"""
knowledge.py -- Extract structured lessons from experiment history.

Reads all results/*.json files and builds a knowledge base that captures:
- Which CONFIG parameter changes correlate with ASR changes
- Which parameter combinations have been tried
- Performance trends over time
- Quick-vs-full fold discrepancies
"""
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

RESULTS_DIR = Path(__file__).parent / "results"
BEST_FILE = Path(__file__).parent / "best.json"
KNOWLEDGE_FILE = Path(__file__).parent / "knowledge.json"


@dataclass
class Lesson:
    """A single extracted lesson from experiment history."""
    category: str              # "param_effect" | "combination" | "failure" | "trend"
    description: str           # Human-readable lesson
    param: Optional[str] = None
    value_from: Optional[str] = None
    value_to: Optional[str] = None
    asr_delta: Optional[float] = None
    confidence: str = "low"    # "high" (5-fold) | "low" (1-fold)
    source_experiments: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """Full knowledge base."""
    lessons: List[Lesson] = field(default_factory=list)
    best_5fold_asr: float = 0.0
    best_5fold_config: Dict = field(default_factory=dict)
    best_5fold_timestamp: str = ""
    best_1fold_asr: float = 0.0
    total_experiments: int = 0
    tried_configs: List[Dict] = field(default_factory=list)
    param_ranges_explored: Dict = field(default_factory=dict)
    quick_to_full_bias: float = 0.04
    consecutive_no_improvement: int = 0
    last_updated: str = ""
    unexplored_param_values: Dict = field(default_factory=dict)


def load_all_results() -> List[Dict]:
    """Load all result JSONs sorted by timestamp."""
    results = []
    for p in sorted(RESULTS_DIR.glob("exp_*.json")):
        try:
            data = json.loads(p.read_text())
            data["_path"] = str(p)
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda r: r.get("timestamp", ""))
    return results


def _config_diff(c1: Dict, c2: Dict) -> Dict[str, Tuple]:
    """Return {param: (old_val, new_val)} for params that differ."""
    diff = {}
    all_keys = set(c1.keys()) | set(c2.keys())
    for k in all_keys:
        v1 = c1.get(k)
        v2 = c2.get(k)
        if v1 != v2:
            diff[k] = (v1, v2)
    return diff


def extract_param_effects(results: List[Dict]) -> List[Lesson]:
    """Compare consecutive experiments to find which param changes caused ASR changes."""
    lessons = []
    seen_transitions = set()

    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        pc = prev.get("config", {})
        cc = curr.get("config", {})
        if not pc or not cc:
            continue

        diff = _config_diff(pc, cc)
        if not diff or len(diff) > 2:
            continue

        prev_asr = prev.get("mean_asr", 0)
        curr_asr = curr.get("mean_asr", 0)
        delta = curr_asr - prev_asr

        is_5fold = curr.get("n_folds", 1) >= 5 and prev.get("n_folds", 1) >= 5
        confidence = "high" if is_5fold else "low"

        for param, (old_val, new_val) in diff.items():
            key = (param, str(old_val), str(new_val))
            if key in seen_transitions:
                continue
            seen_transitions.add(key)

            direction = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
            lessons.append(Lesson(
                category="param_effect",
                description=f"Changing {param} from {old_val} to {new_val} {direction} ASR by {delta:+.4f}",
                param=param,
                value_from=str(old_val),
                value_to=str(new_val),
                asr_delta=round(delta, 4),
                confidence=confidence,
                source_experiments=[prev.get("timestamp", ""), curr.get("timestamp", "")],
            ))
    return lessons


def extract_tried_ranges(results: List[Dict]) -> Dict[str, List]:
    """Build a map of param -> [all values tried]."""
    ranges: Dict[str, set] = {}
    for r in results:
        cfg = r.get("config", {})
        for k, v in cfg.items():
            ranges.setdefault(k, set())
            ranges[k].add(str(v))
    return {k: sorted(list(v)) for k, v in ranges.items()}


def compute_quick_full_gap(results: List[Dict]) -> float:
    """Compute average gap between 1-fold and 5-fold for same config hash."""
    by_hash: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        h = r.get("attack_hash", "")
        nf = r.get("n_folds", 1)
        asr = r.get("mean_asr", 0)
        by_hash.setdefault(h, {"quick": [], "full": []})
        if nf == 1:
            by_hash[h]["quick"].append(asr)
        elif nf >= 5:
            by_hash[h]["full"].append(asr)

    gaps = []
    for h, data in by_hash.items():
        if data["quick"] and data["full"]:
            q_mean = sum(data["quick"]) / len(data["quick"])
            f_mean = sum(data["full"]) / len(data["full"])
            gap = q_mean - f_mean
            if gap > 0:
                gaps.append(gap)

    return round(sum(gaps) / len(gaps), 4) if gaps else 0.04


def _find_best_5fold(results: List[Dict]) -> Tuple[float, Dict, str]:
    """Find best 5-fold result."""
    best_asr = 0.0
    best_cfg = {}
    best_ts = ""
    for r in results:
        if r.get("n_folds", 1) >= 5:
            asr = r.get("mean_asr", 0)
            if asr > best_asr:
                best_asr = asr
                best_cfg = r.get("config", {})
                best_ts = r.get("timestamp", "")
    return best_asr, best_cfg, best_ts


def _find_best_1fold(results: List[Dict]) -> float:
    """Find best 1-fold result."""
    best = 0.0
    for r in results:
        if r.get("n_folds", 1) == 1:
            best = max(best, r.get("mean_asr", 0))
    return best


def _count_consecutive_no_improvement(results: List[Dict], best_5fold_asr: float) -> int:
    """Count how many recent 5-fold experiments didn't beat best."""
    full_results = [r for r in results if r.get("n_folds", 1) >= 5]
    count = 0
    for r in reversed(full_results):
        if r.get("mean_asr", 0) >= best_5fold_asr:
            break
        count += 1
    return count


# Known search space for each parameter
PARAM_SEARCH_SPACE = {
    "feat_scale": ["auto", "auto_x2", "auto_x3", "auto_linear", "1.0", "5.0", "10.0"],
    "sigma": ["1e-05", "0.0001", "0.001", "0.005", "0.01", "0.05", "0.1"],
    "gen_lr": ["0.0001", "0.0005", "0.001", "0.003", "0.005", "0.007", "0.01", "0.02", "0.05"],
    "loss_type": ["cw", "cosine", "hybrid"],
    "edge_strategy": ["topk", "spectral", "full"],
    "kappa": ["-5.0", "-1.0", "-0.5", "-0.1", "-0.01", "-0.001"],
    "node_budget": ["1", "2", "3"],
    "attack_epochs": ["20", "30", "50", "75", "100", "150"],
    "gen_hid_dim": ["64", "128", "256"],
    "grad_method": ["cge", "rgf"],
}


def compute_unexplored(tried: Dict[str, List]) -> Dict[str, List]:
    """Find parameter values not yet tried."""
    unexplored = {}
    for param, candidates in PARAM_SEARCH_SPACE.items():
        tried_vals = set(tried.get(param, []))
        remaining = [v for v in candidates if v not in tried_vals]
        if remaining:
            unexplored[param] = remaining
    return unexplored


def extract_trends(results: List[Dict]) -> List[Lesson]:
    """Extract high-level trends from the experiment history."""
    lessons = []
    full_results = [r for r in results if r.get("n_folds", 1) >= 5]

    if len(full_results) >= 3:
        asrs = [r.get("mean_asr", 0) for r in full_results]
        recent = asrs[-3:]
        if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
            lessons.append(Lesson(
                category="trend",
                description=f"Declining trend in recent 5-fold ASRs: {[round(a, 4) for a in recent]}",
                confidence="high",
            ))
        elif max(recent) - min(recent) < 0.005:
            lessons.append(Lesson(
                category="trend",
                description=f"5-fold ASR plateaued around {sum(recent)/len(recent):.4f}",
                confidence="high",
            ))

    # Check fold variance
    high_var = [r for r in full_results if r.get("std_asr", 0) > 0.08]
    if high_var:
        lessons.append(Lesson(
            category="trend",
            description=f"{len(high_var)} of {len(full_results)} full runs have std > 0.08 — fold variance is a bottleneck",
            confidence="high",
        ))

    return lessons


def build_knowledge_base() -> KnowledgeBase:
    """Main entry point: load results, extract lessons, save to knowledge.json."""
    from datetime import datetime

    results = load_all_results()
    if not results:
        return KnowledgeBase(last_updated=datetime.now().strftime("%Y%m%d_%H%M%S"))

    best_5fold_asr, best_5fold_cfg, best_5fold_ts = _find_best_5fold(results)
    best_1fold_asr = _find_best_1fold(results)
    tried = extract_tried_ranges(results)
    unexplored = compute_unexplored(tried)
    bias = compute_quick_full_gap(results)

    lessons = []
    lessons.extend(extract_param_effects(results))
    lessons.extend(extract_trends(results))

    no_imp = _count_consecutive_no_improvement(results, best_5fold_asr)

    kb = KnowledgeBase(
        lessons=lessons,
        best_5fold_asr=best_5fold_asr,
        best_5fold_config=best_5fold_cfg,
        best_5fold_timestamp=best_5fold_ts,
        best_1fold_asr=best_1fold_asr,
        total_experiments=len(results),
        tried_configs=[r.get("config", {}) for r in results],
        param_ranges_explored=tried,
        quick_to_full_bias=bias,
        consecutive_no_improvement=no_imp,
        last_updated=datetime.now().strftime("%Y%m%d_%H%M%S"),
        unexplored_param_values=unexplored,
    )
    return kb


def save_knowledge(kb: KnowledgeBase):
    """Serialize to knowledge.json."""
    data = {
        "last_updated": kb.last_updated,
        "total_experiments": kb.total_experiments,
        "best_5fold": {
            "asr": kb.best_5fold_asr,
            "config": kb.best_5fold_config,
            "timestamp": kb.best_5fold_timestamp,
        },
        "best_1fold_asr": kb.best_1fold_asr,
        "quick_to_full_bias": kb.quick_to_full_bias,
        "param_ranges_explored": kb.param_ranges_explored,
        "unexplored_param_values": kb.unexplored_param_values,
        "consecutive_no_improvement": kb.consecutive_no_improvement,
        "lessons": [asdict(l) for l in kb.lessons],
    }
    KNOWLEDGE_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_knowledge() -> KnowledgeBase:
    """Load existing knowledge.json."""
    if not KNOWLEDGE_FILE.exists():
        return build_knowledge_base()
    data = json.loads(KNOWLEDGE_FILE.read_text())
    lessons = [Lesson(**l) for l in data.get("lessons", [])]
    b5 = data.get("best_5fold", {})
    return KnowledgeBase(
        lessons=lessons,
        best_5fold_asr=b5.get("asr", 0),
        best_5fold_config=b5.get("config", {}),
        best_5fold_timestamp=b5.get("timestamp", ""),
        best_1fold_asr=data.get("best_1fold_asr", 0),
        total_experiments=data.get("total_experiments", 0),
        param_ranges_explored=data.get("param_ranges_explored", {}),
        quick_to_full_bias=data.get("quick_to_full_bias", 0.04),
        consecutive_no_improvement=data.get("consecutive_no_improvement", 0),
        last_updated=data.get("last_updated", ""),
        unexplored_param_values=data.get("unexplored_param_values", {}),
    )


if __name__ == "__main__":
    kb = build_knowledge_base()
    save_knowledge(kb)
    print(f"Knowledge base built from {kb.total_experiments} experiments")
    print(f"Best 5-fold ASR: {kb.best_5fold_asr:.4f}")
    print(f"Best 1-fold ASR: {kb.best_1fold_asr:.4f}")
    print(f"Quick-to-full bias: {kb.quick_to_full_bias:.4f}")
    print(f"Lessons extracted: {len(kb.lessons)}")
    print(f"Consecutive no-improvement: {kb.consecutive_no_improvement}")
    print(f"\nUnexplored parameter values:")
    for p, vals in kb.unexplored_param_values.items():
        print(f"  {p}: {vals}")
    print(f"\nTop lessons:")
    for l in kb.lessons[:10]:
        print(f"  [{l.confidence}] {l.description}")
