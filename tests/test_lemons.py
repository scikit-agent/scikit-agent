"""The lemons market: adverse selection, and a fixed point an acyclic graph hides."""

import numpy as np
import pytest

import skagent.models.lemons as lemons
from skagent.algos.best_response import TabularBestResponseSolver
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator

SIZE = 50000


def clearing_run(p0, periods, seed=0):
    """Simulate the market from a starting price; each period is one round."""
    sim = Simulator(
        lemons.lemons_calibration(size=SIZE),
        lemons.lemons_block,
        {"S": lemons.seller_rule},
        {"p": p0},
        sample_count=1,
        T_sim=periods,
        seed=seed,
    )
    sim.initialize_sim()
    return sim.simulate()


class TestTheClearingIterationIsASimulation:
    """``p`` is a lag, so simulating T periods runs T rounds of the map."""

    def test_the_price_path_is_the_analytic_contraction(self):
        # If ``p`` were not an arrival state the sellers would face the price
        # their own decisions imply, and no iteration would happen at all.
        path = np.asarray(clearing_run(1.0, 10)["p"]).ravel()
        assert path == pytest.approx(
            [lemons.analytic_price(1.0, t) for t in range(1, 11)], abs=0.02
        )

    def test_the_market_trades_before_it_collapses(self):
        # Without this the collapse below is indistinguishable from a market
        # that never traded.
        history = clearing_run(1.0, 10)
        volume = np.asarray(history["S"]).sum(axis=-1).ravel()
        assert volume[0] > 0.4 * SIZE
        assert volume[-1] < 0.1 * SIZE
        assert (np.asarray(history["u"]) >= 0).all()


class TestNoTradeIsTheEquilibrium:
    """At ``p = 0`` the mask is empty, and that is the fixed point."""

    def test_the_empty_mask_stays_at_zero_rather_than_going_nan(self):
        history = clearing_run(lemons.EQUILIBRIUM_PRICE, 4)
        assert np.asarray(history["p"]).ravel() == pytest.approx(0.0)
        assert not np.isnan(np.asarray(history["p"])).any()
        assert np.asarray(history["S"]).sum() == 0.0


class TestAnAcyclicGraphDoesNotMeanASweepSuffices:
    """The feedback runs through a structural equation, which the graph misses."""

    def test_the_relevance_graph_sees_one_decision_and_no_cycle(self):
        graph = lemons.lemons_block.relevance_graph(lemons.lemons_calibration())
        assert list(graph.nodes()) == ["S"]
        assert list(graph.edges()) == []
        # So the sweep's own refusal, which fires on a component of more than
        # one decision, would not fire here: it would solve S against whatever
        # price it was handed and return a rule inconsistent with the price
        # that rule induces.
        assert all(len(component) == 1 for component in graph.condensation())

    def test_the_wrong_answer_is_latent_behind_the_entity_guard(self):
        # Nothing in the sweep protects this model. What refuses it today is the
        # solver's entity guard, on an unrelated argument about which axis the
        # leading one is -- so retiring that guard exposes the hazard above.
        ground = GroundedBlock(lemons.lemons_block, lemons.lemons_calibration(size=8))
        with pytest.raises(NotImplementedError, match="no equilibrium concept"):
            TabularBestResponseSolver(ground)

    def test_the_market_equation_is_reported_as_a_crossing(self):
        crossings = lemons.lemons_block.crossings()
        assert set(crossings) == {"p"}
        assert {argument for argument, _, _ in crossings["p"]} == {"theta", "S"}
