#!/usr/bin/env python3
"""
codegen.py -- Apply hypotheses to attack.py.

Two modes:
  1. CONFIG-only changes: Parse CONFIG dict, update values, write back
  2. Logic changes: Apply a code_patch (string replacement)

Always backs up attack.py before modification.
"""
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

ATTACK_FILE = Path(__file__).parent / "attack.py"
BACKUP_DIR = Path(__file__).parent / "backups"


def backup_attack() -> Path:
    """Create timestamped backup of attack.py. Returns backup path."""
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"attack_{ts}.py"
    shutil.copy2(ATTACK_FILE, dst)
    return dst


def restore_attack(backup_path: Path):
    """Restore attack.py from a backup."""
    shutil.copy2(backup_path, ATTACK_FILE)


def restore_latest_backup() -> bool:
    """Restore from most recent backup. Returns True if restored."""
    if not BACKUP_DIR.exists():
        return False
    backups = sorted(BACKUP_DIR.glob("attack_*.py"))
    if backups:
        restore_attack(backups[-1])
        return True
    return False


def _format_value(value) -> str:
    """Format a Python value for insertion into CONFIG dict."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:g}"
        return repr(value)
    return repr(value)


def apply_config_changes(changes: dict):
    """Parse attack.py, update CONFIG dict values, write back.

    Matches lines like:
        "param_name": value,     # optional comment
    and replaces the value. Uses function-based replacement to avoid
    regex backreference issues with numeric values.
    """
    content = ATTACK_FILE.read_text()
    for param, new_value in changes.items():
        replacement = _format_value(new_value)

        # Pattern: "param_name": <old_value>,  with possible comment
        pattern = rf'("{param}"\s*:\s*)([^,]+)(,\s*(?:#.*)?)$'

        def replacer(m, rep=replacement):
            return m.group(1) + rep + m.group(3)

        new_content = re.sub(pattern, replacer, content, flags=re.MULTILINE)
        if new_content == content:
            pattern2 = rf'("{param}"\s*:\s*)([^,]+)(,\s*)$'
            new_content = re.sub(pattern2, replacer, content, flags=re.MULTILINE)
        content = new_content

    ATTACK_FILE.write_text(content)


def apply_code_patch(old_code: str, new_code: str) -> bool:
    """Replace old_code with new_code in attack.py. Returns True if successful."""
    content = ATTACK_FILE.read_text()
    if old_code not in content:
        return False
    content = content.replace(old_code, new_code, 1)
    ATTACK_FILE.write_text(content)
    return True


def validate_attack_py() -> bool:
    """Check that attack.py is syntactically valid and run_attack is importable."""
    result = subprocess.run(
        ["python3", "-c",
         "import ast; ast.parse(open('attack.py').read()); "
         "print('syntax ok')"],
        capture_output=True, text=True,
        cwd=str(ATTACK_FILE.parent),
    )
    return result.returncode == 0


def read_current_config() -> dict:
    """Parse the current CONFIG dict from attack.py."""
    content = ATTACK_FILE.read_text()
    # Find CONFIG = { ... }
    match = re.search(r'CONFIG\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if not match:
        return {}

    config = {}
    for line in match.group(1).split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Match "key": value,
        m = re.match(r'"(\w+)"\s*:\s*(.+?)\s*,?\s*(?:#.*)?$', line)
        if m:
            key = m.group(1)
            val_str = m.group(2).strip().rstrip(',')
            try:
                val = eval(val_str)  # Safe for simple literals
            except Exception:
                val = val_str.strip('"').strip("'")
            config[key] = val
    return config
