"""Tests for construction-time backend selection."""

import builtins
import sys

import numpy as np
import pytest


def test_importing_baldr_does_not_import_numpy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"jax", "numpy"} or name.startswith(("jax.", "scipy")):
            raise AssertionError(f"optional dependency imported eagerly: {name}")
        return original_import(name, *args, **kwargs)

    for name in tuple(sys.modules):
        if name == "baldr" or name.startswith("baldr."):
            del sys.modules[name]
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import baldr

    assert baldr.Normal().pdf(0.0) > 0.0


def test_public_api_selects_backend_once():
    import baldr
    from baldr.numpy import Normal as NumPyNormal
    from baldr.scalar import Normal as ScalarNormal

    assert isinstance(baldr.Normal(), ScalarNormal)
    assert isinstance(baldr.Normal(backend="numpy"), NumPyNormal)
    assert isinstance(baldr.Normal(backend="numpy").pdf([0.0]), np.ndarray)


def test_unknown_backend_fails_at_construction():
    import baldr

    with pytest.raises(ValueError, match="backend"):
        baldr.Normal(backend="other")
