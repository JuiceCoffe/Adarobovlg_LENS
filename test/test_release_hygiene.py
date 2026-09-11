"""Check source, example-data, and documentation release hygiene.

Run from the repository root with ``python test/test_release_hygiene.py``. The check reads
file names and metadata only; it never prints values that resemble credentials.
"""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("model", "robots", "dexgraspapi", "utils", "server", "client", "test")
REMOVED_EXAMPLES = {
    "cup_handle.pkl",
    "light_blue_box.pkl",
    "spatula_handle.pkl",
    "tennis_ball.pkl",
    "transparent_bottle.pkl",
    "white_lid.pkl",
    "white_stapler.pkl",
}
PRIVATE_PATTERNS = {
    "absolute home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "internal data path": re.compile(r"/data/[^\s'\"]+"),
}


def sha256(path):
    """Return the SHA256 digest of a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source_privacy():
    """Reject personal paths and identifiers from project-owned text files."""
    failures = []
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for rule, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{path.relative_to(ROOT)}: {rule}")
    assert not failures, "\n".join(failures)


def check_generated_files():
    """Reject generated Python and packaging artifacts from release directories."""
    failures = []
    for directory in SOURCE_DIRS + ("third_party",):
        for path in (ROOT / directory).rglob("*"):
            if path.name == "__pycache__" or path.name.endswith((".pyc", ".egg-info")):
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, "\n".join(failures)


def check_examples():
    """Validate the public manifest, retained files, and checksums."""
    examples = ROOT / "data" / "examples"
    for name in REMOVED_EXAMPLES:
        assert not (examples / "static" / name).exists(), name
    manifest = json.loads((examples / "manifest.json").read_text(encoding="utf-8"))
    allowed_keys = {"file", "prompt", "obs_pose", "sha256"}
    assert manifest, "Example manifest is empty"
    for item in manifest:
        assert set(item) == allowed_keys, item["file"]
        path = examples / item["file"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path


def check_readme_anchors():
    """Require stable anchors and matching structure in both README files."""
    anchors = {
        "pipeline-overview",
        "environment-installation",
        "model-weights",
        "service-usage",
        "adarobovlg-playground-evaluation",
        "directory-structure",
        "third-party-sub-libraries",
        "additional-documentation",
        "acknowledgements",
        "license",
    }
    signatures = []
    for filename in ("README.md", "README_zh-CN.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "](#-" not in text, filename
        for anchor in anchors:
            assert f'<a id="{anchor}"></a>' in text, f"{filename}: {anchor}"
        signatures.append(
            (
                [len(match.group(1)) for match in re.finditer(r"^(#+) ", text, re.MULTILINE)],
                re.findall(r'<a id="([^"]+)"></a>', text),
                re.findall(r"!?\[[^]]*\]\(([^) ]+)", text),
                re.findall(r"^```([^\n]*)$", text, re.MULTILINE),
            )
        )
    assert signatures[0] == signatures[1], "README structure or targets differ"


def check_release_exclusions():
    """Validate release exclusions when the repository provides a gitignore file."""
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return
    patterns = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        ".agents/",
        ".codex/",
        "docs/open_source_anonymization_audit_zh-CN.md",
        "checkpoints/*",
        "!checkpoints/README.md",
        "!checkpoints/manifest.json",
        "data/pointcloud/",
        "**/.thumbs/",
    }
    assert required <= patterns, sorted(required - patterns)


if __name__ == "__main__":
    check_source_privacy()
    check_generated_files()
    check_examples()
    check_readme_anchors()
    check_release_exclusions()
    print("PASS: privacy, generated files, examples, and README alignment")
