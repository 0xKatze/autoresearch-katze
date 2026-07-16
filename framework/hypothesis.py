#!/usr/bin/env python3
"""
hypothesis.py -- Generate experiment hypotheses.

Uses knowledge base + program.md ideas to propose the next experiment.
Strategies:
  1. Exploit: Small tweaks to best-known config
  2. Explore: Try untested parameter combinations
  3. Ablation: Test removing recent additions
"""
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from knowledge import KnowledgeBase


@dataclass
class Hypothesis:
    """A proposed experiment."""
    name: str
    rationale: str
    config_changes: Dict = field(default_factory=dict)
    code_changes: Optional[str] = None   # Description
    code_patch_old: Optional[str] = None  # Old code to replace
    code_patch_new: Optional[str] = None  # New code
    strategy: str = "exploit"   # "exploit" | "explore" | "ablation" | "program_idea"
    expected_asr_delta: float = 0.0
    run_mode: str = "quick"     # "quick" | "full"
    priority: int = 5           # 1=highest


# Perturbation rules for exploit strategy
EXPLOIT_PERTURBATIONS = {
    "sigma": {
        "type": "float_log",
        "factor": [0.5, 2.0, 0.2, 5.0],
    },
    "gen_lr": {
        "type": "float_log",
        "factor": [0.5, 2.0, 0.3, 3.0],
    },
    "kappa": {
        "type": "float_log",
        "factor": [0.1, 10.0, 0.5, 2.0],
    },
    "attack_epochs": {
        "type": "int_add",
        "delta": [-10, 10, 25, 50],
    },
    "gen_hid_dim": {
        "type": "int_set",
        "values": [64, 128, 256, 512],
    },
    "node_budget": {
        "type": "int_set",
        "values": [1, 2, 3],
    },
    "feat_scale": {
        "type": "str_set",
        "values": ["auto", "auto_x2", "auto_x3", "auto_linear"],
    },
    "loss_type": {
        "type": "str_set",
        "values": ["cw", "cosine", "hybrid"],
    },
    "edge_strategy": {
        "type": "str_set",
        "values": ["topk", "spectral", "full"],
    },
    "grad_method": {
        "type": "str_set",
        "values": ["cge", "rgf"],
    },
}


class HypothesisGenerator:
    def __init__(self, knowledge: KnowledgeBase, program_ideas: List[str]):
        self.knowledge = knowledge
        self.program_ideas = program_ideas
        self.force_explore = False
        self._tried_hypotheses: List[str] = []

    def generate_candidates(self) -> List[Hypothesis]:
        """Generate ranked list of hypothesis candidates."""
        candidates = []
        if not self.force_explore:
            candidates.extend(self._exploit_hypotheses())
        candidates.extend(self._explore_hypotheses())
        candidates.extend(self._ablation_hypotheses())
        # Sort by priority (lower = higher priority)
        candidates.sort(key=lambda h: h.priority)
        return candidates

    def select_next(self) -> Hypothesis:
        """Select the single best next hypothesis to test."""
        candidates = self.generate_candidates()
        # Filter already-tried
        for c in candidates:
            if c.name not in self._tried_hypotheses:
                self._tried_hypotheses.append(c.name)
                return c

        # If all tried, reset and try explore
        self._tried_hypotheses.clear()
        self.force_explore = True
        candidates = self.generate_candidates()
        if candidates:
            self._tried_hypotheses.append(candidates[0].name)
            return candidates[0]

        # Fallback: random perturbation of best config
        return self._random_perturbation()

    def _exploit_hypotheses(self) -> List[Hypothesis]:
        """Small perturbations of best config."""
        best_cfg = self.knowledge.best_5fold_config
        if not best_cfg:
            return []

        hypotheses = []
        # Use lessons to find promising directions
        positive_lessons = [l for l in self.knowledge.lessons
                           if l.category == "param_effect"
                           and l.asr_delta is not None
                           and l.asr_delta > 0]

        # For each param in best config, try small perturbations
        for param, rules in EXPLOIT_PERTURBATIONS.items():
            current = best_cfg.get(param)
            if current is None:
                continue

            new_values = self._get_perturbations(param, current, rules)
            tried_vals = set(self.knowledge.param_ranges_explored.get(param, []))

            for new_val in new_values:
                if str(new_val) in tried_vals:
                    continue
                name = f"exploit_{param}_{new_val}"
                hypotheses.append(Hypothesis(
                    name=name,
                    rationale=f"Perturb {param} from {current} to {new_val}",
                    config_changes={param: new_val},
                    strategy="exploit",
                    expected_asr_delta=0.01,
                    run_mode="quick",
                    priority=3,
                ))

        return hypotheses

    def _explore_hypotheses(self) -> List[Hypothesis]:
        """Try parameter values never seen in history."""
        unexplored = self.knowledge.unexplored_param_values
        hypotheses = []

        for param, values in unexplored.items():
            for val in values[:2]:  # Take top 2 unexplored per param
                # Convert string back to appropriate type
                typed_val = self._parse_value(param, val)
                name = f"explore_{param}_{val}"
                hypotheses.append(Hypothesis(
                    name=name,
                    rationale=f"Unexplored: {param}={val}",
                    config_changes={param: typed_val},
                    strategy="explore",
                    expected_asr_delta=0.0,
                    run_mode="quick",
                    priority=4 if not self.force_explore else 1,
                ))

        return hypotheses

    def _ablation_hypotheses(self) -> List[Hypothesis]:
        """Test if components of best config actually help."""
        best_cfg = self.knowledge.best_5fold_config
        if not best_cfg:
            return []

        hypotheses = []
        # Test switching key params to simpler alternatives
        ablations = [
            ("edge_strategy", "topk", "full", "Test if full connectivity is really needed"),
            ("loss_type", "cw", "hybrid", "Test if hybrid loss component helps"),
            ("feat_scale", "auto", "auto_x2", "Test if 2x scaling is needed over 1x"),
        ]

        for param, simple_val, complex_val, rationale in ablations:
            if best_cfg.get(param) == complex_val:
                name = f"ablation_{param}_{simple_val}"
                hypotheses.append(Hypothesis(
                    name=name,
                    rationale=rationale,
                    config_changes={param: simple_val},
                    strategy="ablation",
                    expected_asr_delta=-0.02,
                    run_mode="quick",
                    priority=6,
                ))

        return hypotheses

    def _get_perturbations(self, param: str, current, rules: dict) -> list:
        """Generate perturbation values for a parameter."""
        ptype = rules["type"]
        values = []

        if ptype == "float_log":
            for factor in rules["factor"]:
                if isinstance(current, str):
                    continue
                new_val = round(current * factor, 6)
                if new_val > 0:
                    values.append(new_val)

        elif ptype == "int_add":
            for delta in rules["delta"]:
                if isinstance(current, str):
                    continue
                new_val = max(1, int(current) + delta)
                values.append(new_val)

        elif ptype in ("int_set", "str_set"):
            for v in rules["values"]:
                if v != current:
                    values.append(v)

        return values

    def _parse_value(self, param: str, val_str: str):
        """Convert string value to appropriate Python type."""
        # String params
        if param in ("feat_scale", "loss_type", "edge_strategy", "grad_method"):
            return val_str
        # Try numeric
        try:
            if '.' in val_str or 'e' in val_str.lower():
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str

    def _random_perturbation(self) -> Hypothesis:
        """Fallback: random perturbation of best config."""
        best_cfg = self.knowledge.best_5fold_config
        param = random.choice(list(EXPLOIT_PERTURBATIONS.keys()))
        rules = EXPLOIT_PERTURBATIONS[param]
        current = best_cfg.get(param)

        if current is not None:
            new_values = self._get_perturbations(param, current, rules)
            if new_values:
                new_val = random.choice(new_values)
                return Hypothesis(
                    name=f"random_{param}_{new_val}",
                    rationale=f"Random perturbation: {param} = {new_val}",
                    config_changes={param: new_val},
                    strategy="explore",
                    run_mode="quick",
                    priority=5,
                )

        return Hypothesis(
            name="noop",
            rationale="No changes to try",
            config_changes={},
            run_mode="quick",
            priority=10,
        )


def parse_program_ideas(program_md_path: str) -> List[str]:
    """Extract the numbered ideas from program.md's 'Ideas to Try' section."""
    try:
        content = Path(program_md_path).read_text()
    except OSError:
        return []

    ideas = []
    in_ideas = False
    for line in content.split('\n'):
        if 'Ideas to Try' in line:
            in_ideas = True
            continue
        if in_ideas:
            if line.startswith('#'):
                break
            m = re.match(r'\d+\.\s+\*\*(.+?)\*\*:?\s*(.*)', line)
            if m:
                ideas.append(f"{m.group(1)}: {m.group(2)}")
    return ideas
