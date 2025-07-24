# Getting Started

Welcome to scikit-agent! This guide will help you get started with the basic concepts and functionality.

## Basic concepts

scikit-agent is designed around several key concepts:

### Models and Blocks

Economic models in scikit-agent are built using a block-based system:

- **DBlock**: Dynamic blocks that define state transitions
- **RBlock**: Reward/return blocks for optimization objectives
- **Control**: Variables that agents choose optimally

### Distributions

Probabilistic modeling is central to economic applications:

- **Bernoulli**: For binary outcomes (survival, employment, etc.)
- **Lognormal**: For positive-valued variables
- **MeanOneLogNormal**: Normalized lognormal for shocks

### Grids and Simulation

Dynamic programming and simulation tools:

- **Grid**: For discretizing continuous state spaces
- **Monte Carlo**: For simulating stochastic processes

## Your first model

Here's a simple example to get you started:

```python
from skagent.distributions import MeanOneLogNormal
from skagent.models.consumer import consumption_block, calibration

# Create an income shock distribution
income_shock = MeanOneLogNormal(sigma=0.1)

# Sample from the distribution
samples = income_shock.draw(n=1000)
print(f"Mean income shock: {samples.mean():.3f}")

# Examine the consumption block
print(f"Block name: {consumption_block.name}")
print(f"Calibration: {calibration}")
```

## Next steps

- Explore the [examples gallery](../auto_examples/index) for detailed use cases
- Read the [API reference](../api/index) for complete documentation
- Check out the [user guide](../user_guide) for in-depth tutorials