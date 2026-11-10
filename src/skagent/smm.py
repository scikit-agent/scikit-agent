r"""
Estimating calibration parameters by the simulated method of moments.

A model's calibration holds the numbers that make its equations concrete. This
module goes the other way from simulation: given observations of the model's
variables, it searches for the calibration values under which the model, run
forward, produces data that look like what was observed.

*Look like* is made precise by a vector of **moments**: statistics of a dataset
chosen by the modeler, such as a mean, a variance, a correlation or the share of
observations above a threshold. The estimator picks the parameters
:math:`\theta` that minimize

.. math::
    Q(\theta) = e(\theta)^\top W \, e(\theta), \qquad
    e(\theta) = \frac{1}{S}\sum_{s=1}^{S} m(\tilde x_s(\theta)) - m(x),

where :math:`m(x)` is the moment vector of the observed data, :math:`\tilde x_s`
is the :math:`s`-th simulated dataset and :math:`W` is a weighting matrix. This is
the simulated method of moments of [McFadden1989]_, which [DuffieSingleton1993]_
extended to dynamic models. Nothing here needs the model's likelihood, which is what makes the method usable on
models whose chance nodes are structural equations of continuous variables, or
whose agents' decisions come from solving a game.

Common random numbers
---------------------
Simulation draws random numbers, and the criterion must not move because the
draws moved. Each of the :math:`S` simulated datasets is therefore run under a
seed that is the same at every :math:`\theta`, so the only thing that changes
between two evaluations of :math:`Q` is the parameter vector. This is what makes
the criterion a smooth function of :math:`\theta` where the model is.

Weighting
---------
``weighting="identity"`` gives every moment equal weight, which is only sensible
if the moments are on comparable scales. ``"diagonal"`` weights each moment by
the inverse of its variance across simulations. ``"two_step"`` (the default)
fits once with the diagonal weights, estimates the full covariance matrix
:math:`\Omega` of the moments at that fit, and fits again with
:math:`W = \Omega^{-1}`, which is the efficient choice.

Standard errors
---------------
With :math:`G = \partial \bar m / \partial \theta` the Jacobian of the average
simulated moments, the covariance of the estimate is

.. math::
    \hat V = \Bigl(1 + \tfrac{1}{S}\Bigr)
        (G^\top W G)^{-1} G^\top W \,\Omega\, W G (G^\top W G)^{-1},

where the factor :math:`1 + 1/S` is the price of estimating the model's moments
by simulation. :math:`\Omega` is the covariance of the moment vector across
simulated datasets, so it describes a dataset of the size that was simulated.
Simulate datasets as large as the observed one; otherwise the standard errors are
too small or too large by the ratio of the sizes.

Limits
------
- The standard errors are first-order asymptotic. They rest on the model being
  correctly specified and on :math:`\Omega` being estimated from enough
  simulations. It is estimated from ``omega_sims`` datasets, more than the
  criterion averages, because its inverse is biased upward when it is estimated
  from few: with ten datasets and three moments the standard errors come out
  about half what they should.
- Moments often share a source of noise, which makes :math:`\Omega` singular
  even with many simulations: in the Cournot market of
  :mod:`skagent.models.cournot`, the average price and the average quantity both
  move with the average firm cost alone. Some
  combination of them is then fixed by the model, and the two-step weighting
  gives that combination a very large weight by flooring the eigenvalues of
  :math:`\Omega` rather than discarding them. A pseudo-inverse would discard
  them, which leaves the fit free to wander along a ridge.
- The parameters are identified only if the moments move with them. A
  parameter no moment responds to cannot be estimated, and neither can two that
  move the moments only in the same proportion, so that the data fix a ratio or
  a product of them and nothing else. In the second case the fit lands anywhere
  along that ridge and the sandwich formula still returns small standard errors.
  :attr:`SMMFit.identification` measures how close the moments come to either,
  and below ``weak_identification`` the fit raises
  :class:`WeakIdentificationWarning` and reports ``nan`` standard errors for the
  parameters involved.
- Moments built from thresholds or counts make the criterion a step function of
  the parameters. The default search, Nelder-Mead, uses no derivatives and copes,
  and the Jacobian behind the standard errors is a slope fitted over a span of
  2% either side of the estimate rather than a difference across a small step.

References
----------
.. [McFadden1989] McFadden, D. (1989). "A Method of Simulated Moments for
       Estimation of Discrete Response Models without Numerical Integration."
       Econometrica, 57(5), 995-1026. :doi:`10.2307/1913621`
.. [DuffieSingleton1993] Duffie, D. and Singleton, K. J. (1993). "Simulated
       Moment Estimation of Markov Models of Asset Prices." Econometrica,
       61(4), 929-952. :doi:`10.2307/2951768`
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize

from skagent.simulation.monte_carlo import Simulator

__all__ = [
    "SMMFit",
    "WeakIdentificationWarning",
    "block_simulator",
    "fit_smm",
    "simulated_moments",
]

# Keeps a moment that does not vary across simulations from receiving an
# infinite weight in the diagonal weighting.
_VARIANCE_FLOOR = 1e-12

# A restart of the optimizer is kept only if it lowers the criterion by more than
# this fraction of it. Smaller gains are the simplex settling, not escaping.
_MATERIAL = 1e-4


class WeakIdentificationWarning(UserWarning):
    """The moments cannot tell some of the estimated parameters apart."""


@dataclass(frozen=True)
class SMMFit:
    """
    The result of :func:`fit_smm`.

    Attributes
    ----------
    params : dict of str to float
        The estimated value of each free parameter.
    std_errors : dict of str to float
        Standard errors from the sandwich formula in :mod:`skagent.smm`. ``nan``
        where the parameters are not identified by the moments.
    criterion : float
        The weighted quadratic distance :math:`Q` at the estimate.
    data_moments : ndarray of shape (n_moments,)
        The moment vector of the observed data.
    model_moments : ndarray of shape (n_moments,)
        The average simulated moment vector at the estimate. Compare it with
        ``data_moments``: a fit that leaves a moment far from its target is a
        fit the model cannot make.
    weighting_matrix : ndarray of shape (n_moments, n_moments)
        The weighting matrix the final fit used.
    n_sims : int
        How many simulated datasets were averaged.
    converged : bool
        Whether the optimizer of the final fit reports convergence.
    identification : float
        The smallest singular value of the moments' Jacobian once each of its
        rows and columns is scaled to unit length, between 0 and 1. Near 1 every
        parameter moves the moments in its own direction; at 0 some combination
        of them moves nothing. Below ``weak_identification`` the fit warns and
        reports ``nan`` standard errors for the parameters involved.
    """

    params: dict
    std_errors: dict
    criterion: float
    data_moments: np.ndarray
    model_moments: np.ndarray
    weighting_matrix: np.ndarray
    n_sims: int
    converged: bool
    identification: float


def block_simulator(
    block,
    calibration: Mapping[str, Any],
    dr: Mapping[str, Callable] | Callable[[Mapping[str, Any]], Mapping[str, Callable]],
    initial: Mapping[str, Any] | None = None,
    *,
    sample_count: int,
    T_sim: int = 1,
    variables: Sequence[str] | None = None,
) -> Callable[[Mapping[str, float], int], dict]:
    """
    A simulation function for :func:`fit_smm`, built on the library's simulator.

    Parameters
    ----------
    block : DBlock or RBlock
        The model.
    calibration : dict of str to object
        The full calibration. The parameters being estimated are overridden by
        the values :func:`fit_smm` proposes; everything else is held fixed.
    dr : dict of str to callable, or callable
        The decision rules the simulator runs. Pass a function of the proposed
        calibration, returning such a dict, to let a rule depend on the
        parameters, as a rule derived from a first-order condition does.
    initial : dict of str to object, default=None
        Initial values or distributions of arrival states, as for
        :class:`~skagent.simulation.monte_carlo.Simulator`. None means there are
        no arrival states.
    sample_count : int
        The number of simulated samples. Set it to the size of the observed
        dataset, which is what makes the standard errors right.
    T_sim : int, default=1
        The number of periods per sample.
    variables : list of str, default=None
        The variables of the simulated history to keep. None keeps all of them.

    Returns
    -------
    callable
        ``simulate(theta, seed)``, mapping a dict of proposed parameter values
        and an integer seed to a dict of arrays shaped as the simulator's
        history is: ``(T_sim, sample_count) + entity_shape``.
    """

    def simulate(theta: Mapping[str, float], seed: int) -> dict:
        proposed = {**calibration, **theta}
        rules = dr(proposed) if callable(dr) else dr
        simulator = Simulator(
            proposed,
            block,
            rules,
            {} if initial is None else initial,
            sample_count=sample_count,
            T_sim=T_sim,
            seed=seed,
        )
        simulator.initialize_sim()
        history = simulator.simulate()
        keep = history.keys() if variables is None else variables
        return {name: np.asarray(history[name]) for name in keep}

    return simulate


def _moment_draws(simulate, moments, theta, n_sims, seed):
    """The moment vector of each of *n_sims* simulated datasets, shape (S, R)."""
    draws = [
        np.atleast_1d(np.asarray(moments(simulate(theta, seed + s)), dtype=float))
        for s in range(n_sims)
    ]
    return np.vstack(draws)


def simulated_moments(simulate, moments, theta, n_sims=20, seed=0):
    """
    The average simulated moment vector at *theta*.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, seed)``, as returned by :func:`block_simulator`.
    moments : callable
        Maps a dataset, in the form *simulate* returns, to a 1-D array.
    theta : dict of str to float
        The parameter values to simulate under.
    n_sims : int, default=20
        How many datasets to average. Their seeds are ``seed, seed + 1, ...``.
    seed : int, default=0
        The first seed.

    Returns
    -------
    ndarray of shape (n_moments,)
    """
    return _moment_draws(simulate, moments, theta, n_sims, seed).mean(axis=0)


def _jacobian(mean_moments, x, steps, box=None):
    """
    Jacobian of the mean simulated moments, (R, K), as local least-squares slopes.

    Each column is the slope of a line fitted through the moments at five points,
    ``x`` and two steps either side of it, rather than a difference of two. Where
    the outcomes are counts the moments are step functions of the parameters, and
    a single small difference measures how many observations happened to cross a
    threshold rather than the slope; a fit over a wider span averages that out.
    On a smooth model the slope is exact to second order, as a central difference
    is. Points outside *box* are left out, so a parameter at a bound is
    differentiated from one side.
    """
    center = mean_moments(x)
    columns = []
    for k in range(x.size):
        low, high = (None, None) if box is None else box[k]
        offsets = [0.0]
        values = [center]
        for multiple in (-2, -1, 1, 2):
            offset = multiple * steps[k]
            point = x[k] + offset
            if (low is not None and point < low) or (high is not None and point > high):
                continue
            moved = x.copy()
            moved[k] = point
            offsets.append(offset)
            values.append(mean_moments(moved))
        offsets = np.asarray(offsets)
        values = np.vstack(values)
        if offsets.size < 2:
            columns.append(np.full(center.shape, np.nan))
            continue
        spread = offsets - offsets.mean()
        columns.append(spread @ (values - values.mean(axis=0)) / (spread @ spread))
    return np.column_stack(columns)


def _identification(G):
    """
    How close the moments come to losing track of some parameter combination.

    Each row of the Jacobian is scaled to unit length, so that no moment counts
    for more because of the units it is measured in, and each column likewise,
    so that no parameter does. The smallest singular value of what is left is
    returned, with its right singular vector mapped back to the parameters'
    own units. Zero means some combination of the parameters moves no moment at
    all; a small value means the moments move with it only faintly, and the
    standard errors along it are unreliable.
    """
    if not np.all(np.isfinite(G)):
        return 0.0, np.full(G.shape[1], np.nan)
    rows = np.linalg.norm(G, axis=1)
    G = G[rows > 0] / rows[rows > 0, None]
    lengths = np.linalg.norm(G, axis=0)
    if np.any(lengths == 0):
        direction = (lengths == 0).astype(float)
        return 0.0, direction / np.linalg.norm(direction)
    _, singular, right = np.linalg.svd(G / lengths, full_matrices=False)
    if singular.size < G.shape[1]:
        return 0.0, np.full(G.shape[1], np.nan)
    direction = right[-1] / lengths
    return float(singular[-1]), direction / np.linalg.norm(direction)


def _standard_errors(G, W, omega, n_sims, drop=None):
    """
    Sandwich standard errors, ``nan`` where they do not exist.

    A parameter that moves no moment at all has no standard error, and is left
    out of the sandwich so that it does not take the others' with it. *drop*, a
    direction in parameter space, is a combination the moments cannot see: the
    bread is inverted on the directions orthogonal to it, which is what leaves a
    parameter outside the combination with the standard error it would have had
    if the combination were known. Otherwise a sandwich that is still singular
    reports no standard errors.
    """
    seen = np.isfinite(G).all(axis=0) & (np.abs(G).sum(axis=0) > 0)
    errors = np.full(G.shape[1], np.nan)
    if not seen.any():
        return errors
    lengths = np.linalg.norm(G[:, seen], axis=0)
    G = G[:, seen] / lengths
    A = G.T @ W @ G
    values, vectors = np.linalg.eigh((A + A.T) / 2)
    keep = values > 1e-12 * max(values.max(), 0.0)
    if drop is not None and np.all(np.isfinite(drop)):
        unseen = drop[seen] * lengths
        if np.linalg.norm(unseen) > 0:
            overlap = np.abs(vectors.T @ (unseen / np.linalg.norm(unseen)))
            keep[np.argmax(overlap)] = False
    elif not keep.all():
        return errors
    bread = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    variance = (1.0 + 1.0 / n_sims) * bread @ G.T @ W @ omega @ W @ G @ bread
    diag = np.diag(variance) / lengths**2
    errors[seen] = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    return errors


def _describe(names, direction):
    """A parameter combination as text, largest weight first."""
    order = np.argsort(-np.abs(direction))
    scale = direction[order[0]]
    terms = [
        f"{direction[k] / scale:+.2f}*{names[k]}"
        for k in order
        if abs(direction[k] / scale) >= 0.01
    ]
    return " ".join(terms).lstrip("+")


def _inverse_covariance(omega, floor):
    """
    The inverse of a moment covariance matrix, safe where it is nearly singular.

    Moments of one simulated dataset often share a source of noise, so their
    covariance can be singular, and some combination of them is then fixed by
    the model. Such a combination should carry a very large weight: the model
    says the data must match it exactly. A pseudo-inverse does the opposite and
    gives it none. Here the matrix is standardized to a correlation matrix, its
    eigenvalues are raised to *floor*, and it is inverted, so a direction with
    no noise is weighted 1/*floor* times as heavily as an ordinary one.
    """
    scale = np.sqrt(np.maximum(np.diag(omega), _VARIANCE_FLOOR))
    correlation = omega / np.outer(scale, scale)
    values, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    inverse = (vectors / np.maximum(values, floor)) @ vectors.T
    return inverse / np.outer(scale, scale)


def fit_smm(
    simulate,
    moments,
    data,
    free: Mapping[str, float],
    *,
    n_sims: int = 20,
    seed: int = 0,
    weighting: str = "two_step",
    bounds: Mapping[str, tuple] | None = None,
    method: str = "Nelder-Mead",
    options: Mapping[str, Any] | None = None,
    restarts: int = 3,
    omega_sims: int = 100,
    omega_floor: float = 1e-3,
    fd_step: float = 1e-2,
    weak_identification: float = 5e-2,
) -> SMMFit:
    """
    Estimate parameters by the simulated method of moments.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, seed)``, returning a dataset in the same form as
        *data*. See :func:`block_simulator`. It must be deterministic given its
        seed: that is what holds the random draws fixed across parameter values.
    moments : callable
        Maps a dataset to a 1-D array of statistics. It is applied to *data* and
        to every simulated dataset, so the two are compared like with like.
        There must be at least as many moments as free parameters.
    data : object
        The observed dataset.
    free : dict of str to float
        The parameters to estimate, each with a starting value.
    n_sims : int, default=20
        The number of simulated datasets averaged at each evaluation. More
        simulations make the estimate less noisy and the criterion smoother at
        a proportional cost.
    seed : int, default=0
        The first of the ``n_sims`` consecutive seeds.
    weighting : {"two_step", "diagonal", "identity"}, default="two_step"
        See :mod:`skagent.smm`.
    bounds : dict of str to tuple, default=None
        ``(low, high)`` for any of the free parameters, where None leaves a side
        open. None means no parameter is bounded.
    method : str, default="Nelder-Mead"
        Any :func:`scipy.optimize.minimize` method. The default uses no
        derivatives, which suits a simulated criterion.
    options : dict of str to object, default=None
        Passed to the optimizer. None passes nothing.
    restarts : int, default=3
        How many times to restart the optimizer from its own answer while that
        still lowers the criterion materially, by more than a hundredth of a
        percent. A simplex search can collapse against a
        bound or on a ridge and report convergence at a point that is not a
        minimum; starting again from it spreads the simplex out afresh.
    omega_sims : int, default=100
        How many simulated datasets the moment covariance matrix
        :math:`\\Omega` is estimated from, for the weighting and for the
        standard errors. They are drawn under seeds the fit does not use. An
        efficient weighting inverts this matrix, and the inverse of a covariance
        estimated from few draws is biased upward, which makes the standard
        errors too small; with as few draws as moments it does not exist. So
        it is estimated from more datasets than the criterion averages.
    omega_floor : float, default=1e-3
        The smallest eigenvalue allowed in the moment correlation matrix that
        ``"two_step"`` inverts. A combination of moments with no noise is then
        weighted ``1 / omega_floor`` times as heavily as an ordinary one, rather
        than infinitely or not at all.
    fd_step : float, default=1e-2
        Relative step of the Jacobian behind the standard errors, which is a
        least-squares slope over points up to two steps either side of the
        estimate. A step is this fraction of the parameter's value, or of 1 if
        the value is smaller. Moments of counts or thresholds are step functions
        of the parameters, and a step must be wide enough that many observations
        cross it.
    weak_identification : float, default=5e-2
        The value of :attr:`SMMFit.identification` below which the fit warns
        with :class:`WeakIdentificationWarning` and reports ``nan`` standard
        errors for the parameters that cannot be told apart.

    Returns
    -------
    SMMFit

    Raises
    ------
    ValueError
        If *weighting* is not one of the three, *free* is empty, ``n_sims`` is
        below 1 or ``omega_sims`` below 2, there are fewer moments than free
        parameters, or the observed data's moments are not all finite.
    KeyError
        If *bounds* names a parameter that is not free.

    Warns
    -----
    WeakIdentificationWarning
        If :attr:`SMMFit.identification` is below ``weak_identification``.

    See Also
    --------
    block_simulator : Build *simulate* from a block and a calibration.
    simulated_moments : The average simulated moments at given parameters.
    """
    if weighting not in {"two_step", "diagonal", "identity"}:
        raise ValueError(f"Unknown weighting {weighting!r}.")
    if not free:
        raise ValueError("There must be at least one free parameter.")
    if n_sims < 1:
        raise ValueError("At least one simulation is needed.")
    if omega_sims < 2:
        raise ValueError("At least two simulations are needed to estimate Omega.")

    names = list(free)
    start = np.array([free[name] for name in names], dtype=float)
    m_data = np.atleast_1d(np.asarray(moments(data), dtype=float))
    if m_data.size < len(names):
        raise ValueError(
            f"{m_data.size} moments cannot identify {len(names)} parameters."
        )
    if not np.all(np.isfinite(m_data)):
        raise ValueError("The observed data's moments are not all finite.")
    box = None
    if bounds:
        unknown = [name for name in bounds if name not in free]
        if unknown:
            raise KeyError(f"Bounds given for non-free parameters: {unknown}")
        box = [bounds.get(name, (None, None)) for name in names]

    def draws_at(x):
        return _moment_draws(simulate, moments, dict(zip(names, x)), n_sims, seed)

    def mean_moments(x):
        return draws_at(x).mean(axis=0)

    def covariance_at(x):
        # Drawn under seeds the fit never uses: the estimate is tuned to its
        # own draws, and a covariance taken from them would be too small.
        draws = _moment_draws(
            simulate, moments, dict(zip(names, x)), omega_sims, seed + n_sims
        )
        return np.atleast_2d(np.cov(draws, rowvar=False))

    def criterion(x, W):
        e = mean_moments(x) - m_data
        return float(e @ W @ e)

    def minimize(x0, W, omega):
        # The optimizer's tolerances are absolute, so it sees the criterion in
        # units of its own sampling noise, where a value of about the number of
        # moments is a good fit whatever the weighting matrix. Under the identity
        # weighting the raw criterion can be far below the default tolerance,
        # and the search would stop before it had begun.
        noise = float(np.trace(W @ omega))
        scale = r / noise if noise > 0 else 1.0
        result = None
        for _ in range(restarts + 1):
            attempt = optimize.minimize(
                lambda x: scale * criterion(x, W),
                x0 if result is None else result.x,
                method=method,
                bounds=box,
                options=dict(options or {}),
            )
            if result is not None and attempt.fun >= result.fun - _MATERIAL * (
                1.0 + abs(result.fun)
            ):
                break
            result = attempt
        return result

    r = m_data.size
    omega = covariance_at(start)
    if weighting == "identity":
        W = np.eye(r)
    else:
        W = np.diag(1.0 / np.maximum(np.diag(omega), _VARIANCE_FLOOR))

    result = minimize(start, W, omega)
    if weighting == "two_step":
        omega = covariance_at(result.x)
        W = _inverse_covariance(omega, omega_floor)
        result = minimize(result.x, W, omega)

    x_hat = result.x
    omega = covariance_at(x_hat)
    steps = fd_step * np.maximum(np.abs(x_hat), 1.0)
    G = _jacobian(mean_moments, x_hat, steps, box)
    strength, direction = _identification(G)
    weak = strength < weak_identification
    errors = _standard_errors(G, W, omega, n_sims, direction if weak else None)
    if weak:
        involved = (
            np.ones(len(names), dtype=bool)
            if not np.all(np.isfinite(direction))
            else np.abs(direction) >= 0.1 * np.nanmax(np.abs(direction))
        )
        errors = np.where(involved, np.nan, errors)
        combination = (
            ", ".join(names)
            if not np.all(np.isfinite(direction))
            else _describe(names, direction)
        )
        warnings.warn(
            "The moments barely move along the parameter combination "
            f"{combination} (identification {strength:.2g}), so the data cannot "
            "tell those parameters apart and their estimates are arbitrary along "
            "it. Their standard errors are reported as nan. Add a moment that "
            "responds to them separately, or fix one of them.",
            WeakIdentificationWarning,
            stacklevel=2,
        )

    return SMMFit(
        params=dict(zip(names, map(float, x_hat))),
        std_errors=dict(zip(names, map(float, errors))),
        criterion=criterion(x_hat, W),
        data_moments=m_data,
        model_moments=mean_moments(x_hat),
        weighting_matrix=W,
        n_sims=n_sims,
        converged=bool(result.success),
        identification=strength,
    )
