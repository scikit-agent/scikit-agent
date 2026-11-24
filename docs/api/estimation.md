# Estimation

This section contains the API documentation for estimating a model's calibration
parameters from data.

## Simulated Method of Moments

The estimator chooses the parameter values under which the model, simulated
forward, reproduces chosen statistics of the observed data. It needs no
likelihood. See {mod}`skagent.smm` for the criterion, the standard errors and
the limits.

```{eval-rst}
.. autofunction:: skagent.smm.fit_smm

.. autoclass:: skagent.smm.SMMFit
   :members:

.. autofunction:: skagent.smm.simulated_moments

.. autoclass:: skagent.smm.WeakIdentificationWarning
```

## Example Usage

Estimating the demand intercept and slope of the Cournot market from observed
prices and quantities. Each firm's best response depends on the demand
parameters, so `simulate` rebuilds the decision rule for every proposed value.

```python
import numpy as np

from skagent.models import cournot
from skagent.simulation.monte_carlo import Simulator
from skagent.smm import fit_smm

calibration = cournot.heterogeneous_calibration(size=50)


def simulate(theta, seed):
    A, b = theta["A"], theta["b"]
    rule = {"q": lambda c: (A - c) / (2 * b)}
    simulator = Simulator(
        {**calibration, **theta},
        cournot.cournot_block,
        rule,
        {},
        sample_count=300,  # as many markets as were observed
        T_sim=1,
        seed=seed,
    )
    simulator.initialize_sim()
    history = simulator.simulate()
    return {"P": np.asarray(history["P"]), "Q": np.asarray(history["Q"])}


def moments(data):
    P, Q = data["P"].ravel(), data["Q"].ravel()
    return np.array([P.mean(), Q.mean(), P.var(), Q.var()])


# Here the "observed" data are simulated under A = 10, b = 1.
observed = simulate({"A": 10.0, "b": 1.0}, seed=10_000)

fit = fit_smm(
    simulate,
    moments,
    observed,
    free={"A": 8.0, "b": 2.0},
    bounds={"A": (1.0, None), "b": (0.1, None)},
)
fit.params, fit.std_errors
```
