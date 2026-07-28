"""Package-level smoke tests."""

import baldr


def test_version_is_available() -> None:
    """The package exposes a version without requiring an installed wheel."""

    assert baldr.__version__
