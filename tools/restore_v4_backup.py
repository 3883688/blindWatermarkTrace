"""Restore entry point for an injected guarded V4 initialization workflow."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.initialize_v4 import main


if __name__ == "__main__":
    raise SystemExit(main())
