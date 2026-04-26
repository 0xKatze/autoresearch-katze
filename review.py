#!/usr/bin/env python3
"""
review.py -- Multi-perspective peer review for attack methodology.

Inspired by AutoResearchClaw's multi-agent peer review.
Generates structured reviews from multiple perspectives:
1. Attack Effectiveness Reviewer — focuses on ASR, success patterns
2. Statistical Rigor Reviewer — focuses on evaluation methodology
3. Practical Security Reviewer — focuses on real-world applicability

Output: reviews.md with actionable feedback.
"""
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).parent / "results"
REVIEWS_FILE = Path(__file__).parent / "reviews.md"


@dataclass
class ReviewFinding:
    reviewer: str
    category: str       # "strength" | "weakness" | "suggestion"
    severity: str       # "critical" | "major" | "minor" | "positive"
    finding: str
    evidence: str
    recommendation: str


def _load_results() -> List[Dict]:
    results = []
    for p in sorted(RESULTS_DIR.glob("exp_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda r: r.get("timestamp", ""))
    return results


def _load_best() -> Dict:
    best_path = Path(__file__).parent / "best.json"
    if best_path.exists():
        return json.loads(best_path.read_text())
    return {}


def _review_attack_effectiveness(results: List[Dict], best: Dict) -> List[ReviewFinding]:
    """Reviewer A: Attack Effectiveness"""
    findings = []
    full_results = [r for r in results if r.get("n_folds", 1) >= 5]
    best_asr = best.get("best_asr", 0)
    best_cfg = best.get("best_config", {})

    # Strength: Overall ASR
    if best_asr > 0.9:
        findings.append(ReviewFinding(
            reviewer="Attack Effectiveness",
            category="strength", severity="positive",
            finding=f"Achieved {best_asr:.1%} ASR on 5-fold evaluation",
            evidence=f"Best config: {json.dumps(best_cfg, default=str)}",
            recommendation="This is a strong result; document the attack pipeline clearly",
        ))
    elif best_asr < 0.5:
        findings.append(ReviewFinding(
            reviewer="Attack Effectiveness",
            category="weakness", severity="critical",
            finding=f"ASR of {best_asr:.1%} is below random chance for binary classification",
            evidence=f"Random baseline would achieve ~50% on PROTEINS (2-class)",
            recommendation="Fundamental approach may need rethinking",
        ))

    # Check if attack degrades over time (overfitting to evaluation)
    if len(full_results) >= 5:
        recent_asrs = [r.get("mean_asr", 0) for r in full_results[-5:]]
        if all(recent_asrs[i] <= recent_asrs[0] for i in range(1, len(recent_asrs))):
            findings.append(ReviewFinding(
                reviewer="Attack Effectiveness",
                category="weakness", severity="major",
                finding="ASR declining in recent experiments — possible overfit to hyperparams",
                evidence=f"Recent 5-fold ASRs: {[f'{a:.2%}' for a in recent_asrs]}",
                recommendation="Try a fundamentally different approach rather than more tuning",
            ))

    # Check multi-restart effectiveness
    if best_cfg.get("attack_epochs", 0) > 75:
        findings.append(ReviewFinding(
            reviewer="Attack Effectiveness",
            category="suggestion", severity="minor",
            finding=f"Using {best_cfg.get('attack_epochs')} epochs per graph",
            evidence="More epochs = more queries. Budget efficiency matters for black-box.",
            recommendation="Report query budget alongside ASR for fair comparison",
        ))

    # Edge strategy
    edge = best_cfg.get("edge_strategy", "")
    if edge == "full":
        findings.append(ReviewFinding(
            reviewer="Attack Effectiveness",
            category="suggestion", severity="minor",
            finding="Full connectivity to all nodes may be unrealistic",
            evidence="Real attackers may not know graph structure",
            recommendation="Discuss full vs selective connectivity as a threat model assumption",
        ))

    return findings


def _review_statistical_rigor(results: List[Dict], best: Dict) -> List[ReviewFinding]:
    """Reviewer B: Statistical Rigor"""
    findings = []
    full_results = [r for r in results if r.get("n_folds", 1) >= 5]
    quick_results = [r for r in results if r.get("n_folds", 1) == 1]

    # 5-fold evaluation
    if full_results:
        findings.append(ReviewFinding(
            reviewer="Statistical Rigor",
            category="strength", severity="positive",
            finding=f"Uses {full_results[0].get('n_folds', 5)}-fold cross-validation",
            evidence=f"{len(full_results)} full evaluations conducted",
            recommendation="Good practice; ensures results generalize across data splits",
        ))

    # Quick-vs-full gap
    if quick_results and full_results:
        quick_best = max(r.get("mean_asr", 0) for r in quick_results)
        full_best = max(r.get("mean_asr", 0) for r in full_results)
        gap = quick_best - full_best
        if gap > 0.05:
            findings.append(ReviewFinding(
                reviewer="Statistical Rigor",
                category="weakness", severity="major",
                finding=f"Large gap between 1-fold ({quick_best:.1%}) and 5-fold ({full_best:.1%}) best",
                evidence=f"Gap: {gap:.1%} — suggests 1-fold results are overly optimistic",
                recommendation="Report only 5-fold results; use 1-fold only for screening",
            ))

    # Fold variance
    high_var = [r for r in full_results if r.get("std_asr", 0) > 0.08]
    if high_var:
        findings.append(ReviewFinding(
            reviewer="Statistical Rigor",
            category="weakness", severity="major",
            finding=f"{len(high_var)}/{len(full_results)} experiments have std > 8%",
            evidence=f"Highest std: {max(r.get('std_asr', 0) for r in high_var):.4f}",
            recommendation="Consider reporting median instead of mean, or analyze per-fold failures",
        ))

    # Number of experiments
    if len(results) < 10:
        findings.append(ReviewFinding(
            reviewer="Statistical Rigor",
            category="weakness", severity="minor",
            finding=f"Only {len(results)} experiments conducted",
            evidence="Small sample may not explore hyperparameter space sufficiently",
            recommendation="Run at least 20+ experiments with systematic parameter sweeps",
        ))
    else:
        findings.append(ReviewFinding(
            reviewer="Statistical Rigor",
            category="strength", severity="positive",
            finding=f"{len(results)} experiments conducted with systematic optimization",
            evidence="Sufficient exploration of hyperparameter space",
            recommendation="Include ablation study showing each component's contribution",
        ))

    # Confidence intervals
    if full_results:
        stds = [r.get("std_asr", 0) for r in full_results if r.get("std_asr", 0) > 0]
        if stds:
            avg_std = sum(stds) / len(stds)
            findings.append(ReviewFinding(
                reviewer="Statistical Rigor",
                category="suggestion", severity="minor",
                finding=f"Average fold std: {avg_std:.4f}",
                evidence="Should report 95% CI in final results",
                recommendation=f"95% CI ≈ ±{avg_std * 2.776:.4f} (t-distribution, df=4)",
            ))

    return findings


def _review_practical_security(results: List[Dict], best: Dict) -> List[ReviewFinding]:
    """Reviewer C: Practical Security"""
    findings = []
    best_cfg = best.get("best_config", {})

    # Query budget
    feat_dim = 3  # PROTEINS
    grad = best_cfg.get("grad_method", "cge")
    epochs = best_cfg.get("attack_epochs", 50)
    if grad == "cge":
        queries_per_graph = epochs * 2 * feat_dim
    else:
        queries_per_graph = epochs * 100  # rgf eval_num
    findings.append(ReviewFinding(
        reviewer="Practical Security",
        category="suggestion", severity="minor",
        finding=f"Query budget: ~{queries_per_graph} model queries per graph",
        evidence=f"grad_method={grad}, epochs={epochs}, feat_dim={feat_dim}",
        recommendation="Compare with other black-box attacks' query budgets",
    ))

    # Node budget
    budget = best_cfg.get("node_budget", 1)
    findings.append(ReviewFinding(
        reviewer="Practical Security",
        category="strength" if budget == 1 else "suggestion",
        severity="positive" if budget == 1 else "minor",
        finding=f"Node injection budget: {budget} node(s)",
        evidence="Single node injection is the most constrained (realistic) setting",
        recommendation="Good — demonstrates attack effectiveness with minimal perturbation"
        if budget == 1 else "Consider testing with budget=1 for strongest claims",
    ))

    # Feature scaling
    fs = best_cfg.get("feat_scale", "auto")
    if "x2" in str(fs) or "x3" in str(fs):
        findings.append(ReviewFinding(
            reviewer="Practical Security",
            category="weakness", severity="major",
            finding=f"Feature scaling: {fs} amplifies injected features beyond natural range",
            evidence="Scaled features may be detectable by anomaly detection",
            recommendation="Discuss detectability; test if attack works with feat_scale=auto (1x)",
        ))

    # Dataset limitation
    findings.append(ReviewFinding(
        reviewer="Practical Security",
        category="suggestion", severity="minor",
        finding="Evaluation limited to PROTEINS dataset with GCN victim",
        evidence="Single dataset + single model may not generalize",
        recommendation="Test on ENZYMES, MNIST-S; also test GIN, GraphSAGE victims",
    ))

    return findings


def _methodology_evidence_consistency(results: List[Dict],
                                       best: Dict) -> List[ReviewFinding]:
    """Cross-check: Are the claimed results consistent with the evidence?"""
    findings = []
    best_asr = best.get("best_asr", 0)
    full_results = [r for r in results if r.get("n_folds", 1) >= 5]

    if full_results:
        actual_best_full = max(r.get("mean_asr", 0) for r in full_results)
        # Check if best_asr comes from 1-fold
        if best_asr > actual_best_full + 0.01:
            findings.append(ReviewFinding(
                reviewer="Methodology-Evidence Consistency",
                category="weakness", severity="critical",
                finding=f"Reported best ASR ({best_asr:.1%}) exceeds best 5-fold ({actual_best_full:.1%})",
                evidence="Best ASR appears to come from 1-fold evaluation",
                recommendation="Report best 5-fold ASR as primary result, not 1-fold",
            ))
        else:
            findings.append(ReviewFinding(
                reviewer="Methodology-Evidence Consistency",
                category="strength", severity="positive",
                finding="Reported results are consistent with 5-fold evaluations",
                evidence=f"Best 5-fold: {actual_best_full:.1%}",
                recommendation="Good — results are reproducible",
            ))

    return findings


def generate_review(output_path: Optional[Path] = None) -> Path:
    """Generate full multi-perspective review."""
    results = _load_results()
    best = _load_best()
    out = output_path or REVIEWS_FILE

    all_findings: List[ReviewFinding] = []
    all_findings.extend(_review_attack_effectiveness(results, best))
    all_findings.extend(_review_statistical_rigor(results, best))
    all_findings.extend(_review_practical_security(results, best))
    all_findings.extend(_methodology_evidence_consistency(results, best))

    # Build markdown
    lines = [
        f"# Peer Review Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Experiments reviewed**: {len(results)}",
        f"**Best 5-fold ASR**: {max((r.get('mean_asr', 0) for r in results if r.get('n_folds', 1) >= 5), default=0):.1%}",
        f"",
        f"---",
        f"",
    ]

    # Group by reviewer
    reviewers = {}
    for f in all_findings:
        reviewers.setdefault(f.reviewer, []).append(f)

    for reviewer, findings in reviewers.items():
        lines.append(f"## {reviewer}")
        lines.append("")

        strengths = [f for f in findings if f.category == "strength"]
        weaknesses = [f for f in findings if f.category == "weakness"]
        suggestions = [f for f in findings if f.category == "suggestion"]

        if strengths:
            lines.append("### Strengths")
            for f in strengths:
                lines.append(f"- **{f.finding}**")
                lines.append(f"  - Evidence: {f.evidence}")
                lines.append(f"  - {f.recommendation}")
                lines.append("")

        if weaknesses:
            lines.append("### Weaknesses")
            for f in weaknesses:
                severity_icon = {"critical": "!!!", "major": "!!", "minor": "!"}.get(f.severity, "")
                lines.append(f"- **[{f.severity.upper()}]** {f.finding}")
                lines.append(f"  - Evidence: {f.evidence}")
                lines.append(f"  - Recommendation: {f.recommendation}")
                lines.append("")

        if suggestions:
            lines.append("### Suggestions")
            for f in suggestions:
                lines.append(f"- {f.finding}")
                lines.append(f"  - {f.recommendation}")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Summary
    n_critical = sum(1 for f in all_findings if f.severity == "critical")
    n_major = sum(1 for f in all_findings if f.severity == "major")
    n_positive = sum(1 for f in all_findings if f.severity == "positive")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Category | Count |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Positive findings | {n_positive} |")
    lines.append(f"| Critical issues | {n_critical} |")
    lines.append(f"| Major issues | {n_major} |")
    lines.append(f"| Total findings | {len(all_findings)} |")
    lines.append("")

    if n_critical > 0:
        lines.append("**Action Required**: Address critical issues before reporting results.")
    elif n_major > 0:
        lines.append("**Action Recommended**: Address major issues to strengthen the paper.")
    else:
        lines.append("**Status**: Results are solid. Minor improvements possible.")

    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    path = generate_review()
    print(f"Review written to: {path}")
    print(path.read_text())
