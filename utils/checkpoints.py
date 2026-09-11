"""Resolve local model assets without depending on the user's download cache."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(relative_path, env_var=None, directory=False):
    """Resolve per-model override, then shared root, and report missing local assets.

    Relative paths are interpreted against the project root. Only the requested
    asset is checked so static gripper use does not require tracking/hand weights.
    """
    base = Path(os.environ.get("ADAROBOVLG_CHECKPOINTS_DIR", ROOT / "checkpoints")).expanduser()
    path = Path(os.environ.get(env_var, base / relative_path) if env_var else base / relative_path)
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(
            f"Missing model asset: {path}. See checkpoints/README.md; "
            f"set {env_var or 'ADAROBOVLG_CHECKPOINTS_DIR'} to use an existing download."
        )
    return path
