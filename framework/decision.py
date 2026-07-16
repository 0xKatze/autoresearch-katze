#!/usr/bin/env python3
"""
decision.py -- Decide next action in the research loop.

Decisions:
  - PROMOTE: Quick run was promising, run full 5-fold to confirm
  - REFINE:  Current direction works, make small tweaks
  - PIVOT:   Current direction stalled, try something different
  - STOP:    Target achieved or max iterations reached
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(Enum):
    PROMOTE = "promote"
    REFINE = "refine"
    PIVOT = "pivot"
    STOP = "stop"


@dataclass
class Decision:
    action: Action
    reason: str
    details: Optional[str] = None


class DecisionEngine:
    def __init__(self, target_asr: float = 0.95,
                 max_consecutive_failures: int = 3,
                 max_no_improvement: int = 5):
        self.target_asr = target_asr
        self.max_consecutive_failures = max_consecutive_failures
        self.max_no_improvement = max_no_improvement

    def decide(self, analysis: 'AnalysisReport',
               knowledge: 'KnowledgeBase',
               consecutive_failures: int,
               no_improvement_count: int) -> Decision:
        """Core decision logic."""

        # 1. Target achieved
        if knowledge.best_5fold_asr >= self.target_asr:
            return Decision(Action.STOP,
                f"Target ASR {self.target_asr:.2%} achieved: {knowledge.best_5fold_asr:.2%}")

        # 2. Quick run promising → promote to full 5-fold
        if analysis.is_quick_run and not analysis.suspicious:
            if analysis.estimated_5fold_asr > knowledge.best_5fold_asr:
                return Decision(Action.PROMOTE,
                    f"Quick ASR {analysis.mean_asr:.2%} "
                    f"(est 5-fold: {analysis.estimated_5fold_asr:.2%}) "
                    f"> best {knowledge.best_5fold_asr:.2%}")

        # 3. Too many consecutive failures → pivot
        if consecutive_failures >= self.max_consecutive_failures:
            return Decision(Action.PIVOT,
                f"{consecutive_failures} consecutive failures",
                "Try a fundamentally different approach")

        # 4. No improvement in N experiments → pivot
        if no_improvement_count >= self.max_no_improvement:
            return Decision(Action.PIVOT,
                f"No improvement in {no_improvement_count} experiments",
                "Explore untested parameter space")

        # 5. Last run improved → refine in same direction
        if analysis.asr_delta_vs_best > 0 and not analysis.is_quick_run:
            return Decision(Action.REFINE,
                f"New best! +{analysis.asr_delta_vs_best:.4f}")

        # 6. Default: refine with different parameters
        return Decision(Action.REFINE, "Continue exploring variations")
