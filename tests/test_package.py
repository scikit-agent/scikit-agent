from __future__ import annotations

import importlib.metadata

import skagent as m


def test_version():
    assert importlib.metadata.version("scikit-agent") == m.__version__
