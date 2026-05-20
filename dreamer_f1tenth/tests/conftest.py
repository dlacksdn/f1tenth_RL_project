"""pytest path setup — ensure f110_gym editable + pkg.drivers import."""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

for p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "gym"),
    os.path.join(_PROJECT_ROOT, "pkg", "src"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
