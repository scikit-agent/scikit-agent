# Distributions

The distributions module provides probability distributions used for shocks and
initial conditions. Continuous distributions support both drawing random samples
and discretization into point-mass approximations.

```{note}
`Normal` and `Lognormal` are parameterized by the distribution's mean and
standard deviation in levels, not by the log-space parameters
$(\mu, \sigma)$ familiar from other libraries. A standard deviation of zero
produces a degenerate point mass at the `mean` argument.
```

```{eval-rst}
.. automodule:: skagent.distributions
   :members:
```
