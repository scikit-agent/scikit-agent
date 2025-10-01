"""
Copyright (c) 2025 scikit-agent Team. All rights reserved.

scikit-agent: A scikit-learn compatible toolkit for agent-based modeling.

This package provides tools for creating, solving, and simulating models
using modern computational methods including reinforcement learning, neural networks,
and traditional numerical techniques.
"""

from __future__ import annotations

from ._version import version as __version__

# Core model building blocks
from .model import DBlock, RBlock, Control, Aggregate

# Grid and computational tools
from .grid import Grid, make_grid

# Neural network tools
from .ann import Net, BlockPolicyNet

# Simulation tools
from .simulation.monte_carlo import AgentTypeMonteCarloSimulator, MonteCarloSimulator

# Algorithm imports
from .algos import vbi
from .algos import maliar

# Utility functions
from . import utils
from . import parser

__all__ = [
    "__version__",
    # Core classes
    "DBlock",
    "RBlock",
    "Control",
    "Aggregate",
    # Grid tools
    "Grid",
    "make_grid",
    # Neural networks
    "Net",
    "BlockPolicyNet",
    # Simulation
    "AgentTypeMonteCarloSimulator",
    "MonteCarloSimulator",
    # Modules
    "vbi",
    "maliar",
    "utils",
    "parser",
]
