"""
test_changelog.py — guard against version drift between CHANGELOG.md
and the version strings hardcoded in templates/index.html.

If this test starts failing after a release, you forgot to bump one of
the locations. PR #4 will eliminate this drift entirely via APP_VERSION
env var injection.
"""

import re


def test_changelog_has_latest_version():
    """CHANGELOG.md top entry must be a vX.Y.Z header."""
    with open("CHANGELOG.md", encoding="utf-8") as f:
        content = f.read()
    # First version header after the title
    m = re.search(r"^##\s+v(\d+\.\d+\.\d+)", content, re.MULTILINE)
    assert m, "No v#.#.# version header found in CHANGELOG.md"


def test_index_html_version_matches_changelog():
    """index.html APP_VERSION must match the latest CHANGELOG entry.
    Catches forgotten-to-bump-version mistakes."""
    with open("CHANGELOG.md", encoding="utf-8") as f:
        changelog = f.read()
    with open("templates/index.html", encoding="utf-8") as f:
        index_html = f.read()

    # Latest changelog version
    cl_match = re.search(r"^##\s+v(\d+\.\d+\.\d+)", changelog, re.MULTILINE)
    assert cl_match
    latest = cl_match.group(1)

    # APP_VERSION js const in index.html
    idx_match = re.search(r"APP_VERSION\s*=\s*['\"](\d+\.\d+\.\d+)['\"]", index_html)
    assert idx_match, "APP_VERSION constant not found in index.html"
    in_html = idx_match.group(1)

    assert latest == in_html, (
        f"Version mismatch: CHANGELOG says v{latest}, index.html says v{in_html}. "
        f"Update both, or merge PR #4 to centralize this."
    )
