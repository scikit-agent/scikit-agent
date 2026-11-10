# Estimation

This section contains the API documentation for estimating a model's calibration
parameters from data.

## Simulated Method of Moments

The estimator chooses the calibration values under which the model, simulated
forward, reproduces chosen statistics of the observed data. It needs no
likelihood, so it applies to models whose chance nodes are structural equations
and whose agents' decisions come from solving a game.

```{eval-rst}
.. automodule:: skagent.smm
   :no-members:

.. autofunction:: skagent.smm.fit_smm

.. autoclass:: skagent.smm.SMMFit
   :members:

.. autofunction:: skagent.smm.block_simulator

.. autofunction:: skagent.smm.simulated_moments

.. autoclass:: skagent.smm.WeakIdentificationWarning
```

## Example Usage

Estimating the demand intercept and slope of the Cournot market from observed
prices and quantities. The decision rule follows from the first-order condition
and reads the demand parameters, so it is built from the proposed calibration.

```python
import numpy as np

from skagent.models import cournot
from skagent.smm import block_simulator, fit_smm


def rule(calibration):
    A, b = calibration["A"], calibration["b"]
    return {"q": lambda c: (A - c) / (2 * b)}


simulate = block_simulator(
    cournot.cournot_block,
    cournot.heterogeneous_calibration(size=50),
    rule,
    sample_count=300,  # as many markets as were observed
    variables=["P", "Q"],
)


def moments(data):
    P, Q = data["P"].ravel(), data["Q"].ravel()
    return np.array([P.mean(), Q.mean(), P.var(), Q.var()])


# `observed` has the form `simulate` returns: here, simulated under the truth.
observed = simulate({"A": 10.0, "b": 1.0}, 10_000)

fit = fit_smm(
    simulate,
    moments,
    observed,
    free={"A": 8.0, "b": 2.0},
    bounds={"A": (1.0, None), "b": (0.1, None)},
)
fit.params, fit.std_errors
```
