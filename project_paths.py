from __future__ import annotations

from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """
    Find project root by walking up until we find pyproject.toml (or .git).
    """
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found (pyproject.toml or .git missing).")


ROOT = find_root()

# Core dirs
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

# Create dirs (safe if they already exist)
for d in [
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    EVAL_DIR,
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    DOCS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
