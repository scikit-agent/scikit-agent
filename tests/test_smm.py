"""
This file implements unit tests for the simulated method of moments.
"""

from typing import ClassVar

import numpy as np
import pytest
from scipy.stats import chi2

from skagent.block import DBlock
from skagent.distributions import Uniform
from skagent.models import cournot
from skagent.smm import (
    SMMFit,
    WeakIdentificationWarning,
    _identification,
    _jacobian,
    _standard_errors,
    block_simulator,
    fit_smm,
    simulated_moments,
)

# A linear model with uniform noise. Its moments have closed forms, which is
# what lets a test say what the estimator should find rather than only that it
# finds something.
#
#   x ~ U(0, 1),  u ~ U(-1, 1),  y = a + b x + s u
#   E[y] = a + b/2,  Var[y] = b^2/12 + s^2/3,  Cov[x, y] = b/12
linear_block = DBlock(
    name="linear",
    shocks={
        "x": (Uniform, {"low": 0.0, "high": 1.0}),
        "u": (Uniform, {"low": -1.0, "high": 1.0}),
    },
    dynamics={"y": lambda x, u, a, b, s: a + b * x + s * u},
)

TRUTH = {"a": 1.0, "b": 2.0, "s": 0.5}


def linear_moments(data):
    x, y = data["x"].ravel(), data["y"].ravel()
    return np.array(
        [y.mean(), y.var(), np.cov(x, y)[0, 1], np.abs(y - y.mean()).mean()]
    )


def linear_simulator(n, calibration=None):
    return block_simulator(
        linear_block,
        {**TRUTH, **(calibration or {})},
        {},
        sample_count=n,
        T_sim=1,
        variables=["x", "y"],
    )


def observed(n, seed=10_000):
    """A dataset from the true parameters, under a seed no fit ever uses."""
    return linear_simulator(n)({}, seed)


START = {"a": 0.0, "b": 1.0, "s": 1.0}
BOUNDS = {"s": (0.01, None)}


class TestSimulatedMoments:
    def test_the_same_seed_gives_the_same_moments(self):
        simulate = linear_simulator(500)
        one = simulated_moments(simulate, linear_moments, TRUTH, n_sims=3, seed=7)
        two = simulated_moments(simulate, linear_moments, TRUTH, n_sims=3, seed=7)
        np.testing.assert_array_equal(one, two)

    def test_a_different_seed_gives_different_moments(self):
        simulate = linear_simulator(500)
        one = simulated_moments(simulate, linear_moments, TRUTH, n_sims=3, seed=7)
        two = simulated_moments(simulate, linear_moments, TRUTH, n_sims=3, seed=8)
        assert not np.allclose(one, two)

    def test_moments_converge_to_their_closed_forms(self):
        simulate = linear_simulator(20_000)
        got = simulated_moments(simulate, linear_moments, TRUTH, n_sims=4, seed=0)
        a, b, s = TRUTH["a"], TRUTH["b"], TRUTH["s"]
        assert got[0] == pytest.approx(a + b / 2, abs=0.01)
        assert got[1] == pytest.approx(b**2 / 12 + s**2 / 3, abs=0.01)
        assert got[2] == pytest.approx(b / 12, abs=0.005)

    def test_the_draws_are_held_fixed_across_parameter_values(self):
        # A small change of a parameter must make a small change of the moments.
        # With fresh draws at each value the change would be sampling noise.
        simulate = linear_simulator(500)
        base = simulated_moments(simulate, linear_moments, TRUTH, n_sims=2, seed=3)
        moved = simulated_moments(
            simulate, linear_moments, {**TRUTH, "a": TRUTH["a"] + 1e-6}, 2, 3
        )
        np.testing.assert_allclose(moved - base, [1e-6, 0, 0, 0], atol=1e-8)


class TestFitSMM:
    @pytest.mark.parametrize("weighting", ["identity", "diagonal", "two_step"])
    def test_recovers_parameters_within_sampling_error(self, weighting):
        n = 2_000
        fit = fit_smm(
            linear_simulator(n),
            linear_moments,
            observed(n),
            START,
            bounds=BOUNDS,
            n_sims=5,
            omega_sims=40,
            weighting=weighting,
        )
        assert isinstance(fit, SMMFit)
        for name, true_value in TRUTH.items():
            # Four standard errors, since the observed sample is one draw.
            assert abs(fit.params[name] - true_value) < 4 * fit.std_errors[name], name

    def test_the_fit_matches_the_moments_it_was_given(self):
        n = 2_000
        fit = fit_smm(
            linear_simulator(n),
            linear_moments,
            observed(n),
            START,
            bounds=BOUNDS,
            n_sims=5,
            omega_sims=40,
        )
        np.testing.assert_allclose(fit.model_moments, fit.data_moments, atol=0.05)
        # Under the efficient weighting the criterion, net of simulation noise,
        # is the overidentification statistic: chi-squared with as many degrees
        # of freedom as there are moments to spare.
        spare = fit.data_moments.size - len(fit.params)
        statistic = fit.criterion / (1 + 1 / fit.n_sims)
        assert statistic < chi2.ppf(0.999, spare)

    def test_standard_errors_shrink_with_more_data(self):
        small = fit_smm(
            linear_simulator(500),
            linear_moments,
            observed(500),
            START,
            bounds=BOUNDS,
            n_sims=5,
            omega_sims=40,
        )
        large = fit_smm(
            linear_simulator(4_000),
            linear_moments,
            observed(4_000),
            START,
            bounds=BOUNDS,
            n_sims=5,
            omega_sims=40,
        )
        for name in TRUTH:
            assert large.std_errors[name] < small.std_errors[name]

    def test_a_parameter_the_moments_cannot_see_has_no_standard_error(self):
        # The model does not read ``ghost``, so no moment moves with it.
        calibration = {"ghost": 1.0}
        with pytest.warns(WeakIdentificationWarning, match="ghost"):
            fit = fit_smm(
                linear_simulator(1_000, calibration),
                linear_moments,
                observed(1_000),
                {**START, "ghost": 1.0},
                bounds=BOUNDS,
                n_sims=5,
                omega_sims=40,
            )
        assert np.isnan(fit.std_errors["ghost"])
        assert fit.identification == 0.0
        for name in TRUTH:
            assert np.isfinite(fit.std_errors[name]), name

    def test_a_bound_is_respected(self):
        fit = fit_smm(
            linear_simulator(1_000),
            linear_moments,
            observed(1_000),
            START,
            bounds={"b": (0.0, 1.0), "s": (0.01, None)},
            n_sims=5,
            omega_sims=40,
        )
        assert fit.params["b"] <= 1.0 + 1e-9

    def test_parameters_not_free_are_held_fixed(self):
        fit = fit_smm(
            linear_simulator(1_000),
            linear_moments,
            observed(1_000),
            {"a": 0.0, "b": 1.0},
            n_sims=5,
            omega_sims=40,
        )
        assert list(fit.params) == ["a", "b"]


# Only the product of ``a`` and ``b`` reaches the data.
product_block = DBlock(
    name="product",
    shocks={
        "x": (Uniform, {"low": 0.0, "high": 1.0}),
        "u": (Uniform, {"low": -1.0, "high": 1.0}),
    },
    dynamics={"y": lambda x, u, a, b, s: a * b * x + s * u},
)

# A binary outcome, whose moments are step functions of ``p``.
binary_block = DBlock(
    name="binary",
    shocks={"u": (Uniform, {"low": 0.0, "high": 1.0})},
    dynamics={"y": lambda u, p: (u < p) * 1.0},
)


class TestIdentification:
    def test_a_well_identified_fit_says_so(self):
        fit = fit_smm(
            linear_simulator(1_000),
            linear_moments,
            observed(1_000),
            START,
            bounds=BOUNDS,
            n_sims=5,
            omega_sims=40,
        )
        assert fit.identification > 0.1

    def test_parameters_seen_only_as_a_product_are_flagged(self):
        truth = {"a": 2.0, "b": 1.5, "s": 0.5}

        def simulate(theta, seed):
            return block_simulator(
                product_block,
                {**truth, **theta},
                {},
                sample_count=1_000,
                variables=["x", "y"],
            )(theta, seed)

        with pytest.warns(WeakIdentificationWarning, match="a") as caught:
            fit = fit_smm(
                simulate,
                linear_moments,
                simulate(truth, 10_000),
                {"a": 1.0, "b": 1.0, "s": 1.0},
                bounds={"s": (0.01, None)},
                n_sims=5,
                omega_sims=40,
            )
        assert "b" in str(caught[0].message)
        assert np.isnan(fit.std_errors["a"])
        assert np.isnan(fit.std_errors["b"])
        # The product is what the data fix, and the fit recovers it.
        assert fit.params["a"] * fit.params["b"] == pytest.approx(3.0, rel=0.05)
        # The noise scale is identified, and its standard error is the one it
        # has when the product is all there is to estimate.
        known = fit_smm(
            simulate,
            linear_moments,
            simulate(truth, 10_000),
            {"a": 1.0, "s": 1.0},
            bounds={"s": (0.01, None)},
            n_sims=5,
            omega_sims=40,
        )
        assert fit.std_errors["s"] == pytest.approx(known.std_errors["s"], rel=0.2)

    def test_a_binary_outcome_gets_the_binomial_standard_error(self):
        # The share of ones is a step function of p, and a slope over a narrow
        # step measures the observations that happened to cross it. The
        # standard error must still come out at the binomial one.
        n, p = 2_000, 0.3
        simulate = block_simulator(
            binary_block, {"p": p}, {}, sample_count=n, variables=["y"]
        )
        fit = fit_smm(
            simulate,
            lambda data: np.array([data["y"].mean()]),
            simulate({"p": p}, 10_000),
            {"p": 0.5},
            bounds={"p": (0.0, 1.0)},
            n_sims=20,
        )
        binomial = np.sqrt(p * (1 - p) / n * (1 + 1 / 20))
        assert fit.std_errors["p"] == pytest.approx(binomial, rel=0.2)


class TestEdgeCases:
    """The helpers behind the standard errors, where the fit gives them little."""

    def test_a_parameter_pinned_by_its_bounds_has_no_slope(self):
        # Bounds of zero width leave no point to step to on either side.
        G = _jacobian(
            lambda x: np.array([x[0] + x[1], x[1]]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.1]),
            box=[(None, None), (2.0, 2.0)],
        )
        np.testing.assert_allclose(G[:, 0], [1.0, 0.0])
        assert np.isnan(G[:, 1]).all()
        assert _identification(G)[0] == 0.0

    def test_a_slope_is_one_sided_at_a_bound(self):
        G = _jacobian(
            lambda x: np.array([x[0] ** 2]),
            np.array([1.0]),
            np.array([0.01]),
            box=[(1.0, None)],
        )
        assert G[0, 0] == pytest.approx(2.0, rel=0.05)

    def test_fewer_moving_moments_than_parameters_is_unidentified(self):
        # Two parameters, and only one moment that moves at all.
        G = np.array([[1.0, 2.0], [0.0, 0.0]])
        assert _identification(G)[0] == 0.0

    def test_a_singular_sandwich_gives_no_standard_errors(self):
        G = np.array([[1.0, 1.0], [2.0, 2.0]])
        errors = _standard_errors(G, np.eye(2), np.eye(2), n_sims=5)
        assert np.isnan(errors).all()

    def test_an_unseen_parameter_does_not_take_the_others_with_it(self):
        G = np.array([[1.0, 0.0], [0.0, 0.0]])
        errors = _standard_errors(G, np.eye(2), np.eye(2), n_sims=4)
        assert errors[0] == pytest.approx(np.sqrt(1.25))
        assert np.isnan(errors[1])


class TestInputsAreChecked:
    def fit(self, **kwargs):
        defaults = {
            "simulate": linear_simulator(200),
            "moments": linear_moments,
            "data": observed(200),
            "free": START,
            "n_sims": 3,
        }
        defaults.update(kwargs)
        return fit_smm(**defaults)

    def test_rejects_an_unknown_weighting(self):
        with pytest.raises(ValueError, match="Unknown weighting"):
            self.fit(weighting="best")

    def test_rejects_no_free_parameter(self):
        with pytest.raises(ValueError, match="at least one free"):
            self.fit(free={})

    def test_rejects_no_simulations(self):
        with pytest.raises(ValueError, match="At least one simulation"):
            self.fit(n_sims=0)

    def test_rejects_a_covariance_from_a_single_simulation(self):
        with pytest.raises(ValueError, match="two simulations"):
            self.fit(omega_sims=1)

    def test_rejects_fewer_moments_than_parameters(self):
        with pytest.raises(ValueError, match="cannot identify"):
            self.fit(moments=lambda data: np.array([data["y"].mean()]))

    def test_rejects_non_finite_data_moments(self):
        with pytest.raises(ValueError, match="not all finite"):
            self.fit(moments=lambda data: np.full(4, np.nan))

    def test_rejects_bounds_on_a_parameter_that_is_not_free(self):
        with pytest.raises(KeyError, match="non-free"):
            self.fit(bounds={"zzz": (0.0, 1.0)})


class TestCournot:
    """Estimating demand from market outcomes in the library's own Cournot model.

    Firms draw costs uniform on a known range and play the rule that follows
    from the first-order condition, ``q = (A - c) / (2 b)``. That rule reads the
    demand parameters, so it is built from the proposed calibration. Over many
    firms the average price is ``(A + mean cost) / 2`` and the average quantity
    ``(A - mean cost) / (2 b)``, so the demand intercept and slope are read off
    the two.
    """

    TRUE: ClassVar[dict] = {"A": 10.0, "b": 1.0}
    MARKETS = 150

    @staticmethod
    def rule(calibration):
        A, b = calibration["A"], calibration["b"]
        return {"q": lambda c: (A - c) / (2 * b)}

    @staticmethod
    def moments(data):
        P, Q = data["P"].ravel(), data["Q"].ravel()
        return np.array([P.mean(), Q.mean(), P.var(), Q.var()])

    def simulator(self):
        return block_simulator(
            cournot.cournot_block,
            cournot.heterogeneous_calibration(size=50),
            self.rule,
            sample_count=self.MARKETS,
            T_sim=1,
            variables=["P", "Q"],
        )

    def test_recovers_the_demand_parameters(self):
        # The average price and quantity are both linear in the average firm
        # cost, so their covariance matrix is singular. The two-step weighting
        # inverts it anyway; a pseudo-inverse there drops the direction the
        # model fixes exactly, and the fit settles on a ridge far from the truth
        # (A near 4.4).
        simulate = self.simulator()
        data = simulate(self.TRUE, 10_000)
        fit = fit_smm(
            simulate,
            self.moments,
            data,
            {"A": 8.0, "b": 2.0},
            bounds={"A": (1.0, None), "b": (0.1, None)},
            n_sims=5,
            omega_sims=40,
            weighting="two_step",
        )
        for name, true_value in self.TRUE.items():
            assert abs(fit.params[name] - true_value) < 4 * fit.std_errors[name], name
            assert fit.params[name] == pytest.approx(true_value, rel=0.05)
