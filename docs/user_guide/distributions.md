# Distributions

scikit-agent provides a comprehensive set of probability distributions optimized for economic modeling.

## Overview

All distributions in scikit-agent support both `scipy` and `torch` backends for maximum flexibility:

- **scipy backend**: For traditional statistical computing
- **torch backend**: For neural network integration and GPU acceleration

## Available distributions

### Normal
Standard normal distribution for continuous variables.

### Lognormal
Lognormal distribution for positive-valued variables like income or asset prices.

### MeanOneLogNormal
A specialized lognormal distribution normalized to have mean 1.0, commonly used for multiplicative shocks in economic models.

### Bernoulli
Binary distribution for events like survival, employment status, or success/failure outcomes.

### Uniform
Uniform distribution over a specified interval.

## Basic usage

```python
from skagent.distributions import MeanOneLogNormal, Bernoulli

# Create a mean-one lognormal shock
income_shock = MeanOneLogNormal(sigma=0.1)

# Sample values
samples = income_shock.draw(n=1000)
print(f"Mean: {samples.mean():.3f}")  # Should be close to 1.0

# Create a survival probability
survival = Bernoulli(p=0.98)
alive = survival.draw(n=100)
print(f"Survival rate: {alive.mean():.3f}")
```

## Discretization

Many distributions support discretization for use in dynamic programming:

```python
# Discretize a normal distribution using Gauss-Hermite quadrature
from skagent.distributions import Normal

shock = Normal(mu=0, sigma=0.1)
discrete_shock = shock.discretize(n_points=7)

print(f"Discrete points: {discrete_shock.points}")
print(f"Probabilities: {discrete_shock.weights}")
```

## Backend selection

You can choose the computational backend:

```python
# Use scipy backend (default)
dist_scipy = Normal(mu=0, sigma=1, backend="scipy")

# Use torch backend for GPU acceleration
dist_torch = Normal(mu=0, sigma=1, backend="torch")
```