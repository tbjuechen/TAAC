"""Local wrapper for running the competition baseline unchanged.

Some macOS conda setups load duplicate OpenMP runtimes when importing the pip
PyTorch wheel first. Importing pyarrow before executing the baseline entrypoint
is enough for the local demo environment, while keeping ``src/train.py``
byte-for-byte compatible with the competition baseline.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pyarrow  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
runpy.run_path(str(SRC_DIR / "train.py"), run_name="__main__")
