# User Guide

Welcome to the scikit-agent user guide! This comprehensive guide will help you
get started with building, solving, and simulating economic models using
scikit-agent.

## What is scikit-agent?

scikit-agent is a Python package for agent-based economic modeling that follows
scikit-learn conventions. It provides:

**🧱 Modular Architecture**: Build models using composable "blocks" that
represent different stages or aspects of economic behavior.

**🔬 Multiple Solution Methods**: Choose from value function iteration, neural
networks, and other modern computational approaches.

**📊 Rich Simulation Tools**: Generate synthetic data with powerful Monte Carlo
simulators that handle heterogeneity, aging, and complex dynamics.

**🐍 Pythonic API**: Familiar patterns for Python users with clear, readable
code.

## Getting Started

If you're new to scikit-agent, start here:

- {doc}`installation` - Install scikit-agent and set up your environment
- {doc}`quickstart` - Build and simulate your first model in minutes

## Core Concepts

Learn about the fundamental concepts and components:

- {doc}`blocks` - Understanding model structure and building custom models
- {doc}`simulation` - Monte Carlo simulation and analysis
- {doc}`algorithms` - Solution methods for solving your models
- {doc}`science` - Scientific concepts and references

<!---
## Key Design Principles

### Block-Based Models

Models are composed of **Blocks** that capture:

- **Shocks**: Random variables affecting agents
- **Dynamics**: State transition equations
- **Controls**: Decision variables agents optimize
- **Rewards**: Objective functions to maximize

scikit-agent provides an API for building and reusing model blocks.

### Flexible Calibration

Parameters can be:

- Simple numbers: `"CRRA": 2.0`
- Mathematical expressions: `"utility": "c**(1-gamma)/(1-gamma)"`
- Age-varying sequences: `"income": [1.0, 1.2, 1.1, 0.8]`

--->

## Topics Covered

```{toctree}
:maxdepth: 2

installation
quickstart
blocks
simulation
algorithms
science
```

## Need Help?

- **Examples**: Browse the {doc}`../auto_examples/index` for complete working
  examples
- **API Reference**: See {doc}`../api/index` for detailed class and function
  documentation
- **GitHub Issues**: Report bugs or request features at
  [github.com/scikit-agent/scikit-agent](https://github.com/scikit-agent/scikit-agent)

## Contributing

scikit-agent is an open-source project and welcomes contributions! Whether
you're:

- Reporting bugs
- Suggesting new features
- Contributing code
- Improving documentation
- Sharing examples

See our contribution guidelines in the repository for how to get involved.

---

_Ready to get started? Head to the {doc}`quickstart` guide!_
