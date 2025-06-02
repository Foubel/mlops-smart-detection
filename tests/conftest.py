"""Test fixtures and shared configurations for pytest."""

import sys
from pathlib import Path

# Insert the src/ directory at the beginning of sys.path
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
