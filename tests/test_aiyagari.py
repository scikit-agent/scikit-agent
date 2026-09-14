"""Aiyagari: an aggregate that is an average of the cross-section that reads it."""

import numpy as np
import pytest

import skagent.models.aiyagari as aiyagari
from skagent.simulation.monte_carlo import Simulator

ALPHA, DELTA = aiyagari.CAPITAL_SHARE, aiyagari.DEPRECIATION

# Cash on hand includes a household's whole asset position, so a savings rate
# is easier to choose by the interest rate it implies than by eye. This is the
# rate at which the economy settles at 4%.
RATE = aiyagari.savings_rate_for(0.04)
STATIONARY = aiyagari.stationary_capital(RATE)


def run(size, periods, start=1.0, seed=0, samples=1):
    """Simulate *samples* independent economies from a common start."""
    sim = Simulator(
        aiyagari.aiyagari_calibration(size=size),
        aiyagari.aiyagari_block,
        {"c": aiyagari.savings_rule(RATE)},
        {"a": start},
        sample_count=samples,
        T_sim=periods,
        seed=seed,
    )
    sim.initialize_sim()
    return {symbol: np.asarray(path) for symbol, path in sim.simulate().items()}


def marginal_products(capital):
    """The interest rate and wage at *capital*, written out apart from the model."""
    return ALPHA * capital ** (ALPHA - 1) - DELTA, (1 - ALPHA) * capital**ALPHA


@pytest.fixture(scope="module")
def long_run():
    """One 200-period economy at a thousand households, shared by several tests."""
    return run(size=1000, periods=200)


class TestTheAggregateReachesItsStationaryPoint:
    """The one place in the entity work where a dynamic path's arithmetic is checked."""

    def test_capital_arrives_within_three_percent(self, long_run):
        capital = long_run["K"].ravel()
        assert capital[-1] == pytest.approx(STATIONARY, rel=0.03)

        # It got there rather than starting there: the economy opens at the
        # assets every household was given, under a fifth of the stationary value.
        assert capital[0] == pytest.approx(1.0, abs=0.01)
        assert capital[0] < 0.2 * STATIONARY

    def test_the_prices_are_the_marginal_products_every_period(self, long_run):
        interest, wage = marginal_products(long_run["K"].ravel())

        # The market block's arithmetic, checked at every period's capital. With
        # capital's endpoint pinned by the test above, this pins the endpoint
        # prices too, so they need no tolerance of their own.
        assert long_run["R"].ravel() == pytest.approx(interest, rel=1e-12)
        assert long_run["W"].ravel() == pytest.approx(wage, rel=1e-12)

    def test_the_stationary_prices_are_the_marginal_products_there(self):
        # The closed forms must agree with the marginal products at
        # `stationary_capital` and return the 4% `savings_rate_for` was given.
        interest, wage = marginal_products(STATIONARY)
        analytic = aiyagari.stationary_prices(RATE)
        assert analytic["R"] == pytest.approx(interest, rel=1e-12)
        assert analytic["R"] == pytest.approx(0.04, rel=1e-12)
        assert analytic["W"] == pytest.approx(wage, rel=1e-12)

    def test_it_arrives_from_above_as_well(self):
        # The map is increasing and concave through the origin, so the start
        # decides only how long convergence takes; this slope below one at the
        # stationary point sets the speed of the last stretch.
        assert aiyagari.convergence_rate(RATE) < 1

        # Averaged over eight independent economies rather than read off one.
        # The aggregate is a mean over a finite cross-section, so a single run's
        # stationary capital carries sampling noise that no number of periods
        # removes: run once, six seeds spread this from -3.3% to +3.5%, and
        # averaged over samples they span -0.4% to +1.0% for thirty more
        # milliseconds. Averaging is what the sample axis is for.
        capital = run(size=500, periods=120, start=80.0, samples=8)["K"]
        assert capital[0].mean() > STATIONARY
        assert capital[-1].mean() == pytest.approx(STATIONARY, rel=0.03)


class TestThePathIsTheAggregatesOwnLawOfMotion:
    """Each period is one round of the map, not only the last one."""

    def test_every_round_follows_the_closed_form(self, long_run):
        capital = long_run["K"].ravel()
        endowment = long_run["theta"][:, 0, :].mean(axis=-1)

        # A thousand draws average one only to within about 4%, so the map is
        # given each period's realized average; any tolerance on the map at
        # exactly one is either loose or seed-fragile.
        predicted = aiyagari.capital_map(capital[:-1], RATE, endowment=endowment[:-1])
        assert capital[1:] == pytest.approx(predicted, rel=1e-12)


class TestTheCrossSectionSurvives:
    """The aggregate is a mean over households, and the households are not it."""

    def test_the_aggregate_is_scalar_and_the_households_are_not(self, long_run):
        assert long_run["K"].shape == (200, 1)
        for symbol in ("a", "c", "z", "theta"):
            assert long_run[symbol].shape == (200, 1, 1000)

    def test_households_hold_different_amounts(self, long_run):
        # A mean survives a collapsed cross-section, so the aggregate arriving
        # in the right place is not on its own evidence that the households are
        # distinct. Their assets have to spread.
        assets = long_run["a"][-1].ravel()
        assert assets.std() > 0.1 * assets.mean()


class TestEachSampleIsItsOwnEconomy:
    """A reduction over the households must not reach across the sample axis."""

    def test_the_aggregate_is_the_mean_within_a_sample_every_period(self):
        history = run(size=200, periods=30, samples=4)
        capital = history["K"]
        assets = history["a"]
        assert capital.shape == (30, 4)
        assert assets.shape == (30, 4, 200)

        # The market clears on the assets the households arrived with, which are
        # the ones they left the period before holding. Taken within the sample
        # and never across it.
        assert capital[1:] == pytest.approx(assets[:-1].mean(axis=-1))

    def test_the_economies_stay_apart_and_all_of_them_arrive(self):
        # `test_entities` pins the same property on Cournot, but that model is
        # static and its test runs one period, so it cannot show the aggregate
        # staying per-sample once it feeds an arrival state that feeds the next
        # period's aggregate. This is the first dynamic model where it could go
        # wrong over time rather than at once.
        capital = run(size=1000, periods=120, samples=4)["K"]
        assert len(np.unique(capital[1])) == capital.shape[1]
        assert len(np.unique(capital[-1])) == capital.shape[1]

        # They are four draws of one economy rather than four economies, which
        # is two claims: that they are centred on the same capital, and that
        # they do not drift apart. Asserting instead that every sample lands
        # within a band of K* would be a claim about one economy's sampling
        # noise, which is not what this test is about and is several times
        # noisier.
        assert capital[-1].mean() == pytest.approx(STATIONARY, rel=0.03)
        assert capital[-1].std() < 0.05 * STATIONARY


class TestTheTiming:
    """The market clears on what the households arrived with."""

    def test_assets_are_an_arrival_state_and_capital_is_the_crossing(self):
        assert "a" in aiyagari.aiyagari_block.get_arrival_states()

        crossings = aiyagari.aiyagari_block.crossings()
        assert set(crossings) == {"K"}
        assert {argument for argument, _, _ in crossings["K"]} == {"a"}

    def test_the_households_are_a_population_and_the_market_is_not(self):
        assert aiyagari.aiyagari_block.agent_populations() == {"household": "household"}
        assert set(aiyagari.aiyagari_block.entities()) == {"household"}
