r"""
Estimating calibration parameters by the simulated method of moments.

Given observed data, this module searches for the calibration values under
which a model, simulated forward, produces data that look like the observations.
*Look like* is made precise by **moments**: statistics of a dataset chosen by the
modeler, such as a mean or a variance. The estimate :math:`\hat\theta` minimizes

.. math::
    Q(\theta) = e(\theta)^\top W \, e(\theta), \qquad
    e(\theta) = \frac{1}{S}\sum_{s=1}^{S} m(\tilde x_s(\theta)) - m(x),

where :math:`m(x)` is the moment vector of the observed data, :math:`\tilde x_s`
is the :math:`s`-th of :math:`S` simulated datasets and :math:`W` is a diagonal
matrix holding the inverse variance of each moment, measured by simulation at a
first estimate. This is the simulated method
of moments of McFadden (1989). It needs only the ability to simulate the model,
not its likelihood.

Each simulated dataset uses the same seed at every :math:`\theta` (common random
numbers), so the criterion changes only because the parameters changed.

Standard errors use the sandwich formula

.. math::
    \hat V = \Bigl(1 + \tfrac{1}{S}\Bigr)
        (G^\top W G)^{-1} G^\top W \,\Omega\, W G (G^\top W G)^{-1},

with :math:`G` the derivative of the average simulated moments with respect to
:math:`\theta` and :math:`\Omega` the covariance of the moments across simulated
datasets. They are only correct when each simulated dataset is as large as the
observed one.

Limits
------
- The standard errors are first-order asymptotic and assume the model is
  correctly specified.
- The derivative :math:`G` is a central difference. Moments built from counts
  or thresholds change in steps, and the difference can then be noisy; a larger
  ``fd_step`` smooths it. At an estimate on a bound, the difference steps past
  the bound.

References
----------
McFadden, D. (1989). "A Method of Simulated Moments for Estimation of Discrete
Response Models without Numerical Integration." Econometrica, 57(5), 995-1026.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = ["SMMFit", "WeakIdentificationWarning", "fit_smm", "simulated_moments"]


class WeakIdentificationWarning(UserWarning):
    """The moments cannot tell the estimated parameters apart."""


@dataclass(frozen=True)
class SMMFit:
    """
    The result of :func:`fit_smm`.

    Attributes
    ----------
    params : dict of str to float
        The estimated value of each free parameter.
    std_errors : dict of str to float
        The standard error of each estimate.
    criterion : float
        The value of the criterion :math:`Q` at the estimate.
    data_moments : ndarray
        The moments of the observed data.
    model_moments : ndarray
        The average simulated moments at the estimate.
    converged : bool
        Whether the optimizer reports convergence.
    identification : float
        Between 0 and 1. Near 1, each parameter moves the moments in its own
        direction; near 0, two or more of them move the moments alike and the
        data cannot tell them apart.
    """

    params: dict
    std_errors: dict
    criterion: float
    data_moments: np.ndarray
    model_moments: np.ndarray
    converged: bool
    identification: float


def _moment_draws(simulate, moments, theta, n_sims, seed):
    """The moments of each simulated dataset, one row per seed."""
    return np.array(
        [moments(simulate(theta, seed + s)) for s in range(n_sims)], dtype=float
    )


def simulated_moments(simulate, moments, theta, n_sims=20, seed=0):
    """
    The average moments of *n_sims* datasets simulated at *theta*.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, seed)`` returns a dataset simulated at the parameter
        values in the dict *theta*.
    moments : callable
        Maps a dataset to a 1-D array of moments.
    theta : dict of str to float
        The parameter values to simulate at.
    n_sims : int, default=20
        How many datasets to average. Their seeds are ``seed, seed + 1, ...``.
    seed : int, default=0
        The first seed.

    Returns
    -------
    ndarray
    """
    return _moment_draws(simulate, moments, theta, n_sims, seed).mean(axis=0)


def fit_smm(
    simulate,
    moments,
    data,
    free,
    *,
    n_sims=20,
    seed=0,
    bounds=None,
    omega_sims=100,
    fd_step=1e-2,
    weak_identification=5e-2,
):
    """
    Estimate parameters by the simulated method of moments.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, seed)`` returns a dataset simulated at the parameter
        values in the dict *theta*. Everything the model needs other than the
        free parameters is fixed inside it.
    moments : callable
        Maps a dataset to a 1-D array of moments.
    data : object
        The observed dataset, in the form *simulate* returns.
    free : dict of str to float
        The parameters to estimate, with their starting values.
    n_sims : int, default=20
        How many simulated datasets the criterion averages.
    seed : int, default=0
        The first seed of the simulated datasets.
    bounds : dict of str to (float or None, float or None), default=None
        Lower and upper bounds for some of the free parameters.
    omega_sims : int, default=100
        How many extra datasets are simulated to estimate the moments'
        covariance, under seeds the criterion does not use.
    fd_step : float, default=1e-2
        Relative step of the numerical derivative behind the standard errors.
    weak_identification : float, default=5e-2
        Below this value of :attr:`SMMFit.identification` the fit warns and
        reports ``nan`` standard errors.

    Returns
    -------
    SMMFit

    Raises
    ------
    ValueError
        If there are fewer moments than free parameters.
    KeyError
        If *bounds* names a parameter that is not free.

    Warns
    -----
    WeakIdentificationWarning
        If the moments cannot tell the free parameters apart.
    """
    names = list(free)
    start = np.array([free[name] for name in names], dtype=float)
    data_moments = np.atleast_1d(np.asarray(moments(data), dtype=float))
    if data_moments.size < len(names):
        raise ValueError(
            f"{data_moments.size} moments cannot identify {len(names)} parameters."
        )
    box = None
    if bounds:
        unknown = [name for name in bounds if name not in free]
        if unknown:
            raise KeyError(f"Bounds given for parameters that are not free: {unknown}")
        box = [bounds.get(name, (None, None)) for name in names]

    def model_moments(x):
        return simulated_moments(simulate, moments, dict(zip(names, x)), n_sims, seed)

    def moment_covariance(x):
        # Separate seeds: the fit is tuned to its own draws, so their spread
        # would understate the noise.
        draws = _moment_draws(
            simulate, moments, dict(zip(names, x)), omega_sims, seed + n_sims
        )
        return np.atleast_2d(np.cov(draws, rowvar=False))

    def weights_at(x):
        # Weight each moment by the inverse of its variance, so moments on large
        # scales do not dominate the criterion.
        return np.diag(1.0 / np.maximum(np.diag(moment_covariance(x)), 1e-12))

    def criterion(x, W):
        e = model_moments(x) - data_moments
        return float(e @ W @ e)

    def search(x0, W):
        # Nelder-Mead needs no derivatives, which suits a criterion computed by
        # simulation.
        return optimize.minimize(
            criterion, x0, args=(W,), method="Nelder-Mead", bounds=box
        )

    # The variances depend on the parameters. Measured at the starting guess
    # they would make the estimate depend on the guess, so they are measured
    # again at a first estimate and the search is repeated from there.
    result = search(start, weights_at(start))
    W = weights_at(result.x)
    result = search(result.x, W)
    estimate = result.x

    # G: how each moment moves with each parameter, by central differences.
    steps = fd_step * np.maximum(np.abs(estimate), 1.0)
    G = np.column_stack(
        [
            (
                model_moments(estimate + step * unit)
                - model_moments(estimate - step * unit)
            )
            / (2 * step)
            for step, unit in zip(steps, np.eye(len(names)))
        ]
    )
    # If two parameters move the moments in nearly the same direction, the data
    # cannot tell them apart, and the sandwich formula would still return small,
    # meaningless standard errors. Measure the angle between the directions:
    # the smallest singular value of G, with each column scaled to unit length,
    # is 1 when they are perpendicular and 0 when they coincide.
    scaled = np.sqrt(W) @ G
    lengths = np.linalg.norm(scaled, axis=0)
    if np.all(lengths > 0):
        singular = np.linalg.svd(scaled / lengths, compute_uv=False)
        identification = float(singular.min())
    else:
        identification = 0.0

    if identification < weak_identification:
        warnings.warn(
            "The moments cannot tell the free parameters apart (identification "
            f"{identification:.2g}). The standard errors are reported as nan. "
            "Add a moment that responds to them separately, or fix one of them.",
            WeakIdentificationWarning,
            stacklevel=2,
        )
        errors = np.full(len(names), np.nan)
    else:
        omega = moment_covariance(estimate)
        bread = np.linalg.inv(G.T @ W @ G)
        V = (1 + 1 / n_sims) * bread @ G.T @ W @ omega @ W @ G @ bread
        errors = np.sqrt(np.diag(V))

    return SMMFit(
        params=dict(zip(names, map(float, estimate))),
        std_errors=dict(zip(names, map(float, errors))),
        criterion=criterion(estimate, W),
        data_moments=data_moments,
        model_moments=model_moments(estimate),
        converged=bool(result.success),
        identification=identification,
    )
