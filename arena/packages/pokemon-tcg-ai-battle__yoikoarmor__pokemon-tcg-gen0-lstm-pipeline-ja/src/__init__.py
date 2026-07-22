"""Project package. Ensures the bundled `cg` engine module is importable."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE = os.path.join(_ROOT, "sample_submission")
if _SAMPLE not in sys.path:
    sys.path.insert(0, _SAMPLE)
