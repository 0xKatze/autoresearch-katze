"""_paths.py — framework/research path split (added 2026-05-25).

The autoresearch FRAMEWORK lives in framework/; the RESEARCH it drives lives in
research/branch-diffusion/ (attack.py, run.py, prepare.py, results/, charts/, best.json,
knowledge.json, evolution/, backups/, program.md, …). Framework modules reference research
artifacts through RESEARCH_DIR so the framework stays generic and the research stays separable.

Override with the RESEARCH_DIR env var to point the framework at a different research line.
"""
import os
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = Path(os.environ.get(
    "RESEARCH_DIR", FRAMEWORK_DIR.parent / "research" / "branch-diffusion")).resolve()
