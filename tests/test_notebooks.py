"""Structural tests for the committed example notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = (
    "distribution-performance.ipynb",
    "numerical-agreement.ipynb",
)


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_notebook_is_valid_and_code_cells_parse(filename: str) -> None:
    """Check notebook metadata and Python syntax.

    Parameters
    ----------
    filename : str
        Notebook filename below the repository notebook directory.
    """

    path = Path("notebooks") / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{filename}:cell-{index}")
