from __future__ import annotations

import importlib.metadata

import scikit_agent as m


def test_version():
    assert importlib.metadata.version("scikit_agent") == m.__version__
