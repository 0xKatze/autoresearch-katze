#!/usr/bin/env python3
"""
analyzer.py -- Statistical analysis + 4-layer result verification.

Inspired by AutoResearchClaw's 4-layer citation verification, adapted for
experiment result integrity:
  L1: Value range check (ASR in [0,1], std >= 0, wall_time > 0)
  L2: Cross-fold consistency (per-fold ASR variance, outlier detection)
  L3: Historical consistency (compare with similar configs in history)
  L4: Methodology-evidence alignment (quick vs full, config vs ASR)
"""
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VerifyStatus(Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"


@dataclass
class VerificationLayer:
    layer: str           # "L1_range" | "L2_crossfold" | "L3_historical" | "L4_alignment"
    status: VerifyStatus
    detail: str


@dataclass
class VerificationReport:
    layers: List[VerificationLayer] = field(default_factory=list)
    overall_status: VerifyStatus = VerifyStatus.VERIFIED
    integrity_score: float = 1.0  # 0-1, fraction of layers verified

    def summary(self) -> str:
        lines = [f"Integrity: {self.integrity_score:.0%} ({self.overall_status.value})"]
        for l in self.layers:
            icon = {"verified": "✓", "suspicious": "?", "invalid": "✗"}[l.status.value]
            lines.append(f"  {icon} {l.layer}: {l.detail}")
        return "\n".join(lines)


@dataclass
class AnalysisReport:
    """Analysis of a single experiment result."""
    experiment_ts: str
    mean_asr: float
    std_asr: float
    n_folds: int
    is_new_best: bool
    asr_delta_vs_best: float
    asr_delta_vs_previous: float
    is_quick_run: bool
    estimated_5fold_asr: float
    fold_variance_flag: bool       # True if std > 0.08
    config_changes_from_best: Dict
    suspicious: bool
    suspicious_reason: Optional[str]
    wall_time: float
    verification: Optional[VerificationReport] = None


class ResultAnalyzer:
    def __init__(self, knowledge: 'KnowledgeBase'):
        self.knowledge = knowledge

    def analyze(self, result: dict, hypothesis=None) -> AnalysisReport:
        """Analyze a single experiment result."""
        mean_asr = result.get("mean_asr", 0)
        std_asr = result.get("std_asr", 0)
        n_folds = result.get("n_folds", 1)
        is_quick = n_folds == 1
        ts = result.get("timestamp", "")
        wall_time = result.get("wall_time_s", 0)

        # Estimate 5-fold from quick
        if is_quick:
            est_5fold = self._estimate_5fold_from_quick(mean_asr)
        else:
            est_5fold = mean_asr

        # Compare with best
        best_5fold = self.knowledge.best_5fold_asr
        if is_quick:
            is_new_best = False  # Never declare best from 1-fold
            delta_vs_best = est_5fold - best_5fold
        else:
            is_new_best = mean_asr > best_5fold
            delta_vs_best = mean_asr - best_5fold

        # Config diff from best
        best_cfg = self.knowledge.best_5fold_config
        curr_cfg = result.get("config", {})
        config_diff = {}
        for k in set(list(best_cfg.keys()) + list(curr_cfg.keys())):
            if best_cfg.get(k) != curr_cfg.get(k):
                config_diff[k] = {"from": best_cfg.get(k), "to": curr_cfg.get(k)}

        # Previous experiment comparison
        delta_vs_prev = 0.0  # Will be computed by pipeline from last result

        # Quality checks
        suspicious, reason = self._check_suspicious(result)

        # 4-layer verification
        verification = self.verify_result(result)
        if verification.overall_status == VerifyStatus.INVALID:
            suspicious = True
            reason = reason or verification.layers[0].detail

        return AnalysisReport(
            experiment_ts=ts,
            mean_asr=mean_asr,
            std_asr=std_asr,
            n_folds=n_folds,
            is_new_best=is_new_best,
            asr_delta_vs_best=delta_vs_best,
            asr_delta_vs_previous=delta_vs_prev,
            is_quick_run=is_quick,
            estimated_5fold_asr=est_5fold,
            fold_variance_flag=std_asr > 0.08,
            config_changes_from_best=config_diff,
            suspicious=suspicious,
            suspicious_reason=reason,
            wall_time=wall_time,
            verification=verification,
        )

    def _estimate_5fold_from_quick(self, quick_asr: float) -> float:
        """Estimate 5-fold ASR from 1-fold result.

        Uses the empirically observed bias from knowledge base.
        At very high 1-fold ASR (>0.97), gap increases.
        """
        bias = self.knowledge.quick_to_full_bias
        if quick_asr > 0.97:
            bias = max(bias, 0.05)
        return max(0.0, quick_asr - bias)

    def _check_suspicious(self, result: dict) -> tuple:
        """Quality sentinel: flag suspicious results."""
        mean_asr = result.get("mean_asr", 0)
        std_asr = result.get("std_asr", 0)
        n_folds = result.get("n_folds", 1)
        wall_time = result.get("wall_time_s", 0)

        # NaN check
        if math.isnan(mean_asr) or math.isinf(mean_asr):
            return True, "NaN/Inf in ASR"

        # 100% on 1-fold is suspicious
        if n_folds == 1 and mean_asr >= 1.0:
            return True, "100% ASR on 1-fold likely overfit to that fold"

        # Very high std on 5-fold
        if n_folds >= 5 and std_asr > 0.10:
            return True, f"Very high fold variance: std={std_asr:.4f}"

        # Abnormally fast (< 2s for a fold) may indicate error
        if n_folds >= 5 and wall_time < 10:
            return True, f"Suspiciously fast: {wall_time:.1f}s for {n_folds} folds"

        # ASR of 0
        if mean_asr == 0:
            return True, "Zero ASR — attack likely broken"

        return False, None

    def verify_result(self, result: dict) -> VerificationReport:
        """4-layer result verification (inspired by AutoResearchClaw citation verify)."""
        layers = []

        # L1: Value range check
        layers.append(self._verify_L1_range(result))

        # L2: Cross-fold consistency
        layers.append(self._verify_L2_crossfold(result))

        # L3: Historical consistency
        layers.append(self._verify_L3_historical(result))

        # L4: Methodology-evidence alignment
        layers.append(self._verify_L4_alignment(result))

        # Overall
        statuses = [l.status for l in layers]
        if VerifyStatus.INVALID in statuses:
            overall = VerifyStatus.INVALID
        elif VerifyStatus.SUSPICIOUS in statuses:
            overall = VerifyStatus.SUSPICIOUS
        else:
            overall = VerifyStatus.VERIFIED

        verified_count = sum(1 for s in statuses if s == VerifyStatus.VERIFIED)
        integrity = verified_count / len(layers) if layers else 0

        return VerificationReport(
            layers=layers, overall_status=overall, integrity_score=integrity)

    def _verify_L1_range(self, result: dict) -> VerificationLayer:
        """L1: Basic value range validation."""
        asr = result.get("mean_asr", -1)
        std = result.get("std_asr", -1)
        wall = result.get("wall_time_s", -1)

        issues = []
        if not (0 <= asr <= 1):
            issues.append(f"ASR={asr} out of [0,1]")
        if math.isnan(asr) or math.isinf(asr):
            issues.append("ASR is NaN/Inf")
        if std < 0:
            issues.append(f"std={std} negative")
        if wall <= 0:
            issues.append(f"wall_time={wall} non-positive")

        if issues:
            return VerificationLayer("L1_range", VerifyStatus.INVALID, "; ".join(issues))
        return VerificationLayer("L1_range", VerifyStatus.VERIFIED, "All values in valid range")

    def _verify_L2_crossfold(self, result: dict) -> VerificationLayer:
        """L2: Cross-fold consistency check."""
        per_fold = result.get("per_fold_asr", [])
        n_folds = result.get("n_folds", 1)

        if n_folds == 1 or len(per_fold) < 2:
            return VerificationLayer("L2_crossfold", VerifyStatus.VERIFIED,
                                      f"Single fold — no cross-fold check needed")

        # Check for outlier folds (> 2 std from mean)
        mean = sum(per_fold) / len(per_fold)
        std = math.sqrt(sum((x - mean) ** 2 for x in per_fold) / len(per_fold))

        outliers = [i for i, x in enumerate(per_fold) if abs(x - mean) > 2 * std] if std > 0 else []

        if outliers:
            return VerificationLayer("L2_crossfold", VerifyStatus.SUSPICIOUS,
                f"Fold outliers at indices {outliers}: values={[per_fold[i] for i in outliers]}")

        # Check if all folds are identical (suspicious — may indicate bug)
        if len(set(round(x, 6) for x in per_fold)) == 1 and n_folds > 1:
            return VerificationLayer("L2_crossfold", VerifyStatus.SUSPICIOUS,
                "All folds have identical ASR — possible evaluation bug")

        return VerificationLayer("L2_crossfold", VerifyStatus.VERIFIED,
            f"Fold ASRs consistent: mean={mean:.4f}, std={std:.4f}")

    def _verify_L3_historical(self, result: dict) -> VerificationLayer:
        """L3: Compare with similar configs in history."""
        cfg = result.get("config", {})
        asr = result.get("mean_asr", 0)

        # Find experiments with same config in history
        similar = []
        for hist_cfg in self.knowledge.tried_configs:
            if hist_cfg == cfg:
                similar.append(hist_cfg)

        if not similar:
            return VerificationLayer("L3_historical", VerifyStatus.VERIFIED,
                "New config — no historical comparison available")

        # Check if ASR is within expected range
        best_5fold = self.knowledge.best_5fold_asr
        if asr > best_5fold + 0.15:
            return VerificationLayer("L3_historical", VerifyStatus.SUSPICIOUS,
                f"ASR={asr:.4f} is 15pp+ above best historical 5-fold ({best_5fold:.4f})")

        return VerificationLayer("L3_historical", VerifyStatus.VERIFIED,
            f"ASR within historical range (best 5-fold: {best_5fold:.4f})")

    def _verify_L4_alignment(self, result: dict) -> VerificationLayer:
        """L4: Methodology-evidence alignment."""
        asr = result.get("mean_asr", 0)
        n_folds = result.get("n_folds", 1)
        n_attacked = result.get("total_attacked", 0)
        n_success = result.get("total_success", 0)

        issues = []

        # Check success count matches ASR
        if n_attacked > 0:
            computed_asr = n_success / n_attacked
            if abs(computed_asr - asr) > 0.02:
                issues.append(f"ASR mismatch: reported={asr:.4f}, computed={computed_asr:.4f}")

        # Check if 1-fold result being treated as definitive
        if n_folds == 1 and asr >= 0.95:
            issues.append("High ASR on 1-fold needs 5-fold confirmation")

        if issues:
            return VerificationLayer("L4_alignment", VerifyStatus.SUSPICIOUS,
                "; ".join(issues))

        return VerificationLayer("L4_alignment", VerifyStatus.VERIFIED,
            "Methodology and evidence are consistent")
