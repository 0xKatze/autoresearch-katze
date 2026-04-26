#!/usr/bin/env python3
"""
executor.py -- Run experiments with self-healing.

Wraps `python run.py` with:
  - Timeout detection (configurable, default 300s per fold)
  - NaN/Inf detection in results
  - Import/syntax error detection
  - Auto-revert on any failure
"""
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent / "results"
BASE_DIR = Path(__file__).parent


def _detect_python_cmd() -> list:
    """Detect whether to use micromamba or plain python."""
    # Check if we're already in the right conda env
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "graph_adversarial":
        return ["python3"]
    # Check if micromamba is available
    try:
        subprocess.run(["micromamba", "--version"], capture_output=True, timeout=5)
        return ["micromamba", "run", "-n", "graph_adversarial", "python3"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ["python3"]


@dataclass
class ExecutionResult:
    """Result of an experiment execution."""
    success: bool
    result_path: Optional[Path] = None
    result_data: Optional[dict] = None
    error_type: Optional[str] = None   # "timeout" | "runtime" | "nan" | "syntax"
    error_message: Optional[str] = None
    wall_time: float = 0.0
    reverted: bool = False


def run_experiment(quick: bool = True,
                   timeout_s: int = 300,
                   backup_path: Optional[Path] = None) -> ExecutionResult:
    """Run python run.py with self-healing."""
    python_cmd = _detect_python_cmd()
    cmd = python_cmd + ["run.py"]
    if quick:
        cmd.append("--quick")

    # Record existing result files to find the new one
    existing = set(RESULTS_DIR.glob("exp_*.json"))

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout_s,
            cwd=str(BASE_DIR),
        )
        wall = time.time() - t0

        if proc.returncode != 0:
            error_type = _classify_error(proc.stderr + proc.stdout)
            if backup_path:
                _revert(backup_path)
            return ExecutionResult(
                success=False,
                error_type=error_type,
                error_message=(proc.stderr + proc.stdout)[-1000:],
                wall_time=wall,
                reverted=backup_path is not None,
            )

        # Find new result file
        result_path = _find_new_result(existing)
        if result_path is None:
            if backup_path:
                _revert(backup_path)
            return ExecutionResult(
                success=False,
                error_type="no_result",
                error_message="No new result file created",
                wall_time=wall,
                reverted=backup_path is not None,
            )

        result_data = json.loads(result_path.read_text())

        # Validate: check for NaN/Inf
        if _has_nan(result_data):
            if backup_path:
                _revert(backup_path)
            return ExecutionResult(
                success=False,
                result_path=result_path,
                result_data=result_data,
                error_type="nan",
                error_message="NaN/Inf in results",
                wall_time=wall,
                reverted=backup_path is not None,
            )

        return ExecutionResult(
            success=True,
            result_path=result_path,
            result_data=result_data,
            wall_time=wall,
        )

    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        if backup_path:
            _revert(backup_path)
        return ExecutionResult(
            success=False,
            error_type="timeout",
            error_message=f"Exceeded {timeout_s}s",
            wall_time=wall,
            reverted=backup_path is not None,
        )


def _find_new_result(existing: set) -> Optional[Path]:
    """Find the result file that didn't exist before."""
    current = set(RESULTS_DIR.glob("exp_*.json"))
    new_files = current - existing
    if not new_files:
        return None
    return max(new_files, key=lambda p: p.stat().st_mtime)


def _has_nan(result_data: dict) -> bool:
    asr = result_data.get("mean_asr", 0)
    return math.isnan(asr) or math.isinf(asr)


def _classify_error(output: str) -> str:
    if "SyntaxError" in output:
        return "syntax"
    if "ImportError" in output or "ModuleNotFoundError" in output:
        return "import"
    if "RuntimeError" in output and "CUDA" in output:
        return "cuda"
    if "MemoryError" in output or "OOM" in output:
        return "memory"
    return "runtime"


def _revert(backup_path: Path):
    from codegen import restore_attack
    restore_attack(backup_path)
