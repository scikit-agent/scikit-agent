import numpy as np
import pytest

from skagent.models import cournot
from skagent.simulation.monte_carlo import Simulator
from skagent.smm import WeakIdentificationWarning, fit_smm, simulated_moments

# A linear model, y = a + b x + s e, whose moments are known in closed form:
# E[y] = a, Cov(x, y) = b and Var(y) = b^2 + s^2.
TRUTH = {"a": 1.0, "b": 2.0, "s": 0.5}
N = 1000


def linear_simulate(theta, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    y = theta["a"] + theta["b"] * x + theta["s"] * rng.normal(size=N)
    return {"x": x, "y": y}


def linear_moments(data):
    x, y = data["x"], data["y"]
    return np.array([y.mean(), np.cov(x, y)[0, 1], y.var()])


@pytest.fixture(scope="module")
def linear_fit():
    data = linear_simulate(TRUTH, seed=10_000)
    return fit_smm(
        linear_simulate,
        linear_moments,
        data,
        free={"a": 0.0, "b": 1.0, "s": 1.0},
        bounds={"s": (0.01, None)},
    )


def test_simulated_moments_match_the_closed_form():
    m = simulated_moments(linear_simulate, linear_moments, TRUTH, n_sims=50)
    expected = [1.0, 2.0, 2.0**2 + 0.5**2]
    np.testing.assert_allclose(m, expected, atol=0.05)


def test_the_same_seed_gives_the_same_moments():
    first = simulated_moments(linear_simulate, linear_moments, TRUTH, n_sims=3)
    again = simulated_moments(linear_simulate, linear_moments, TRUTH, n_sims=3)
    other = simulated_moments(linear_simulate, linear_moments, TRUTH, n_sims=3, seed=7)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_recovers_the_truth_within_its_standard_errors(linear_fit):
    assert linear_fit.converged
    for name, value in TRUTH.items():
        assert abs(linear_fit.params[name] - value) < 3 * linear_fit.std_errors[name]


def test_standard_errors_have_the_size_of_the_sampling_noise(linear_fit):
    # The mean of N draws of y has standard deviation sqrt(Var(y) / N).
    expected = np.sqrt((2.0**2 + 0.5**2) / N)
    assert 0.5 * expected < linear_fit.std_errors["a"] < 2 * expected


def test_the_model_moments_match_the_data_moments_at_the_estimate(linear_fit):
    np.testing.assert_allclose(
        linear_fit.model_moments, linear_fit.data_moments, rtol=0.05
    )


def test_the_estimate_does_not_depend_on_the_starting_guess(linear_fit):
    far = fit_smm(
        linear_simulate,
        linear_moments,
        linear_simulate(TRUTH, seed=10_000),
        free={"a": -20.0, "b": 15.0, "s": 0.01},
        bounds={"s": (0.01, None)},
    )
    for name in TRUTH:
        assert far.params[name] == pytest.approx(linear_fit.params[name], abs=0.01)


def test_parameters_the_moments_cannot_separate_get_nan_errors():
    # Only the product p * q enters the model, so the data fix the product and
    # nothing else.
    def simulate(theta, seed):
        return linear_simulate({"a": 1.0, "b": theta["p"] * theta["q"], "s": 0.5}, seed)

    with pytest.warns(WeakIdentificationWarning):
        fit = fit_smm(
            simulate,
            linear_moments,
            linear_simulate(TRUTH, seed=10_000),
            free={"p": 1.0, "q": 1.0},
        )
    assert fit.identification < 0.05
    assert all(np.isnan(e) for e in fit.std_errors.values())


def test_a_parameter_that_moves_no_moment_gets_nan_errors():
    def simulate(theta, seed):
        return linear_simulate({"a": theta["a"], "b": theta["b"], "s": 0.5}, seed)

    with pytest.warns(WeakIdentificationWarning):
        fit = fit_smm(
            simulate,
            linear_moments,
            linear_simulate(TRUTH, seed=10_000),
            free={"a": 0.0, "b": 1.0, "unused": 3.0},
        )
    assert fit.identification == 0.0


def test_identified_parameters_do_not_warn(linear_fit):
    assert linear_fit.identification > 0.05


def test_bounds_on_a_parameter_that_is_not_free_is_an_error():
    with pytest.raises(KeyError, match="not free"):
        fit_smm(
            linear_simulate,
            linear_moments,
            linear_simulate(TRUTH, seed=1),
            free={"a": 0.0, "b": 1.0, "s": 1.0},
            bounds={"sigma": (0.0, None)},
        )


def test_fewer_moments_than_parameters_is_an_error():
    with pytest.raises(ValueError, match="cannot identify"):
        fit_smm(
            linear_simulate,
            lambda data: np.array([data["y"].mean()]),
            linear_simulate(TRUTH, seed=1),
            free={"a": 0.0, "b": 1.0},
        )


def test_estimates_cournot_demand_from_prices_and_quantities():
    calibration = cournot.heterogeneous_calibration(size=50)

    def simulate(theta, seed):
        A, b = theta["A"], theta["b"]
        # Each firm's best response depends on the demand parameters, so the
        # decision rule is rebuilt for every proposed value.
        rule = {"q": lambda c: (A - c) / (2 * b)}
        simulator = Simulator(
            {**calibration, **theta},
            cournot.cournot_block,
            rule,
            {},
            sample_count=300,
            T_sim=1,
            seed=seed,
        )
        simulator.initialize_sim()
        history = simulator.simulate()
        return {"P": np.asarray(history["P"]), "Q": np.asarray(history["Q"])}

    def moments(data):
        P, Q = data["P"].ravel(), data["Q"].ravel()
        return np.array([P.mean(), Q.mean(), P.var(), Q.var()])

    fit = fit_smm(
        simulate,
        moments,
        simulate({"A": 10.0, "b": 1.0}, seed=10_000),
        free={"A": 8.0, "b": 2.0},
        bounds={"A": (1.0, None), "b": (0.1, None)},
        n_sims=5,
        omega_sims=40,
    )
    assert fit.params["A"] == pytest.approx(10.0, abs=0.05)
    assert fit.params["b"] == pytest.approx(1.0, abs=0.05)
