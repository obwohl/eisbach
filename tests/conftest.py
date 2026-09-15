"""Make the repository root importable, so `main.py` can be tested.

`pip install -e .` installs the `eisbach` package and nothing else — `packages.find`
includes `eisbach*` only, and `main.py` is an entrypoint script rather than part of the
library, which is right: nothing should be able to `import main` from an installed
checkout. But the entrypoint is also where the orchestration lives, including the `try`
that keeps a failing TimesFM candidate from taking the forecast down, and that is worth a
test.

Running `python -m pytest` from the root papers over this, because `-m` prepends the
working directory to `sys.path`. CI runs bare `pytest -q`, which does not, so the
difference showed up only there. Putting the root on the path here makes both invocations
behave the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
