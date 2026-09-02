#!/usr/bin/env python3
"""Stable entry point for the frozen, resumable main experiment."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_main_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
