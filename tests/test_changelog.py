"""
test_changelog.py — guard against version drift.

After PR #4: version is read at runtime from the VERSION file (or APP_VERSION
env var). The template uses {{ app_version }} so there's no string to drift.
The remaining check: VERSION file should match the latest CHANGELOG entry.
"""

import re
from pathlib import Path


def test_changelog_has_latest_version():
    """CHANGELOG.md top entry must be a vX.Y.Z header."""
    with open("CHANGELOG.md", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"^##\s+v(\d+\.\d+\.\d+)", content, re.MULTILINE)
    assert m, "No v#.#.# version header found in CHANGELOG.md"


def test_version_file_matches_changelog():
    """VERSION file must match the latest CHANGELOG entry.
    This is the single source of truth post-PR-#4."""
    with open("CHANGELOG.md", encoding="utf-8") as f:
        changelog = f.read()
    cl_match = re.search(r"^##\s+v(\d+\.\d+\.\d+)", changelog, re.MULTILINE)
    assert cl_match
    latest = cl_match.group(1)

    version_file = Path("VERSION").read_text().strip()
    assert version_file == latest, (
        f"Version mismatch: CHANGELOG top entry is v{latest}, "
        f"VERSION file says {version_file}. Bump VERSION when you add a CHANGELOG entry."
    )


def test_index_html_uses_template_variable():
    """After PR #4: index.html must NOT hardcode the version anywhere.
    Use {{ app_version }} so there's only one place to bump."""
    with open("templates/index.html", encoding="utf-8") as f:
        index_html = f.read()

    # Should NOT find a hardcoded version like '1.8.1' or v1.8.0
    hardcoded = re.findall(r"APP_VERSION\s*=\s*['\"](\d+\.\d+\.\d+)['\"]", index_html)
    assert not hardcoded, (
        f"Hardcoded version(s) {hardcoded} found in index.html. "
        f"Use {{{{ app_version }}}} instead."
    )
