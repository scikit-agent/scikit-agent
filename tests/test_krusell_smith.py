"""Krusell-Smith: the Aiyagari economy with aggregate productivity shocks."""

import numpy as np
import pytest

import skagent.models.krusell_smith as ks
from skagent.models.aiyagari import savings_rate_for
from skagent.simulation.monte_carlo import Simulator

# At zero aggregate productivity the fixed-rule map is Aiyagari's, so its
# inversion picks a rate that keeps capital in a sensible range.
RATE = savings_rate_for(0.04)


@pytest.fixture(scope="module")
def history():
    """Four economies of 300 households over 60 periods."""
    sim = Simulator(
        ks.krusell_smith_calibration(size=300),
        ks.krusell_smith_block,
        {"c": ks.savings_rule(RATE)},
        {"a": 5.0, "y": 0.0, "Z": 0.0},
        sample_count=4,
        T_sim=60,
        seed=0,
    )
    sim.initialize_sim()
    return {symbol: np.asarray(path) for symbol, path in sim.simulate().items()}


class TestTheAggregatesLawOfMotion:
    def test_every_round_follows_the_closed_form(self, history):
        capital, productivity = history["K"], history["Z"]

        # The endowment is scaled within the class, so the map holds exactly in
        # every period of every economy, not only in expectation.
        predicted = ks.capital_map(capital[:-1], productivity[:-1], RATE)
        assert capital[1:] == pytest.approx(predicted, rel=1e-12)

    def test_productivity_moves_the_prices(self, history):
        # Without this the test above would pass on a market that ignored Z.
        assert history["Z"].std() > 0.01
        alpha, delta = ks.CAPITAL_SHARE, ks.DEPRECIATION
        capital, scale = history["K"], np.exp(history["Z"])
        assert history["R"] == pytest.approx(
            scale * alpha * capital ** (alpha - 1) - delta, rel=1e-12
        )
        assert history["W"] == pytest.approx(
            scale * (1 - alpha) * capital**alpha, rel=1e-12
        )


class TestTheShocks:
    def test_aggregate_productivity_is_one_draw_per_economy(self, history):
        assert history["Z"].shape == (60, 4)
        assert len(np.unique(history["Z"][-1])) == 4

    def test_each_period_opens_with_its_own_innovations(self, history):
        # This period's draws move this period's productivity, which the
        # household then decides on: the decision sees every draw it depends on.
        y, eps_y = history["y"], history["eps_y"]
        assert y[1:] == pytest.approx(0.9 * y[:-1] + eps_y[1:], abs=1e-12)

        Z, eps_Z = history["Z"], history["eps_Z"]
        assert Z[1:] == pytest.approx(0.95 * Z[:-1] + eps_Z[1:], abs=1e-12)

    def test_the_endowment_averages_exactly_one_and_still_varies(self, history):
        theta = history["theta"]
        assert theta.mean(axis=-1) == pytest.approx(1.0, rel=1e-12)
        assert theta[-1].std(axis=-1).min() > 0.1


class TestTheStructure:
    def test_the_arrival_states_and_the_crossings(self):
        calibration = ks.krusell_smith_calibration()
        assert set(ks.krusell_smith_block.get_arrival_states(calibration)) == {
            "a",
            "y",
            "Z",
        }

        crossings = ks.krusell_smith_block.crossings()
        assert {k: {arg for arg, _, _ in v} for k, v in crossings.items()} == {
            "K": {"a"},
            "Theta": {"y"},
        }

    def test_one_household_class_across_two_blocks(self):
        assert set(ks.krusell_smith_block.entities()) == {"household"}
        assert ks.krusell_smith_block.agent_populations() == {"household": "household"}
