"""Check all service imports in fresh subprocesses.

The check imports service dependencies, forwards ``sys.path`` to a fresh interpreter,
and verifies that ``utils.mesh`` resolves to this project rather than a vendored module.
Each subprocess loads the entrypoint as ``__mp_main__`` without starting a server or UI.

Run from the repository root:
    python test/test_server_imports.py

The full project environment is required, but recorded data, weights, GPU inference, and
VLM access are not. Each successful import prints PASS; inference is not evaluated."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # Importing the model dependencies adds Contact-GraspNet paths. A fresh
    # child has these paths but does not inherit the parent's cached utils.
    """Run the test server imports command-line entrypoint."""
    from server.static_grasp import server_dexhand

    assert server_dexhand.ROOT == ROOT
    child_code = """
import json
import runpy
import sys
from pathlib import Path

sys.path = json.loads(sys.stdin.read())
runpy.run_path(sys.argv[1], run_name="__mp_main__")
import utils.mesh
assert Path(utils.mesh.__file__).resolve() == Path(sys.argv[2]) / "utils/mesh.py"
"""
    entries = list((ROOT / "server").glob("*_grasp/server_*.py"))
    entries.extend((ROOT / "server" / "libero").glob("server_*.py"))
    for entry in sorted(entries):
        result = subprocess.run(
            [sys.executable, "-c", child_code, str(entry), str(ROOT)],
            input=json.dumps(sys.path),
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(f"PASS: {entry.relative_to(ROOT)} child imports")


if __name__ == "__main__":
    main()
