"""The lemons market: adverse selection, and a fixed point an acyclic graph hides."""

import numpy as np
import pytest
import torch

import skagent.models.lemons as lemons
from skagent.algos.best_response import TabularBestResponseSolver
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator

SIZE = 50000


def naive_run(p0, periods, seed=0, **configuration):
    """Simulate the posted-price market from *p0*; each period is one round."""
    sim = Simulator(
        lemons.lemons_calibration(size=SIZE, **configuration),
        lemons.naive_lemons_block,
        {"S": lemons.seller_rule},
        {"p": p0},
        sample_count=1,
        T_sim=periods,
        seed=seed,
    )
    sim.initialize_sim()
    return sim.simulate()


def anticipated_run(price, seed=0, **configuration):
    """Play the anticipated-price market once, with sellers expecting *price*."""
    sim = Simulator(
        lemons.lemons_calibration(size=SIZE, **configuration),
        lemons.lemons_block,
        {"S": lemons.supply_rule(price)},
        {},
        sample_count=1,
        T_sim=1,
        seed=seed,
    )
    sim.initialize_sim()
    return sim.simulate()


def buyer_run(price, seed=0, **configuration):
    """Play the monopsony market once at *price*, and return the buyer's surplus."""
    sim = Simulator(
        lemons.lemons_calibration(size=SIZE, **configuration),
        lemons.monopsony_block,
        {"p": lemons.bid_rule(price), "S": lemons.seller_rule},
        {},
        sample_count=1,
        T_sim=1,
        seed=seed,
    )
    sim.initialize_sim()
    return np.asarray(sim.simulate()["w"]).ravel()[0]


class TestTheClearingPriceRunsOnEveryPath:
    """It is a weighted mean, so no path has to special-case it."""

    def test_it_agrees_with_a_mean_over_the_items_that_sold(self):
        theta = np.random.default_rng(0).uniform(0, 1, 5000)
        for price in (0.3, 0.75, 1.0):
            S = lemons.seller_rule(theta, price)
            sold = theta[S > 0.5]
            assert lemons.clearing_price(theta, S, 1.5) == pytest.approx(
                1.5 * sold.mean()
            )

    def test_it_differentiates_and_batches_under_torch(self):
        # A mean over a selection cannot do either: the index is data-dependent,
        # so the shape is dynamic and the gradient in the decision is zero.
        theta = torch.rand(500, dtype=torch.float64)
        S = torch.full((500,), 0.5, dtype=torch.float64, requires_grad=True)
        lemons.clearing_price(theta, S, 1.5).backward()
        assert torch.isfinite(S.grad).all()
        assert S.grad.abs().max() > 0

        batched = torch.vmap(lambda s: lemons.clearing_price(theta, s, 1.5))
        assert batched(torch.rand(8, 500, dtype=torch.float64)).shape == (8,)

    def test_an_empty_pool_prices_at_zero_with_a_gradient_intact(self):
        theta = torch.rand(500, dtype=torch.float64)
        S = torch.zeros(500, dtype=torch.float64, requires_grad=True)
        price = lemons.clearing_price(theta, S, 1.5)
        price.backward()
        assert price.item() == pytest.approx(0.0)
        assert torch.isfinite(S.grad).all()


class TestThePostedPriceIterationIsASimulation:
    """``p`` is a lag, so simulating T periods runs T rounds of the map."""

    def test_the_price_path_is_the_analytic_contraction(self):
        # If ``p`` were not an arrival state the sellers would face the price
        # their own decisions imply, and no iteration would happen at all.
        path = np.asarray(naive_run(1.0, 10)["p"]).ravel()
        assert path == pytest.approx(lemons.clearing_path(1.0, 10), abs=0.02)

    def test_the_market_trades_before_it_collapses(self):
        # Without this the collapse below is indistinguishable from a market
        # that never traded.
        history = naive_run(1.0, 10)
        volume = np.asarray(history["S"]).sum(axis=-1).ravel()
        assert volume[0] > 0.4 * SIZE
        assert volume[-1] < 0.1 * SIZE
        assert (np.asarray(history["u"]) >= 0).all()


class TestAnticipationMakesItAFixedPointInRules:
    """Nothing is lagged, so the equilibrium has to be solved for rather than run."""

    @pytest.mark.parametrize("anticipated", [1.0, 0.6, 0.3])
    def test_the_price_a_rule_induces_is_the_clearing_map_of_the_price_it_expects(
        self, anticipated
    ):
        induced = np.asarray(anticipated_run(anticipated)["p"]).ravel()[0]
        assert induced == pytest.approx(lemons.clearing_map(anticipated), abs=0.01)

    def test_the_equilibrium_rule_reproduces_the_price_it_anticipates(self):
        # Which is the whole content of the equilibrium: at any other price the
        # sellers are acting on a number the market does not go on to produce.
        market = lemons.MARKETS["partial-collapse"]
        equilibrium = lemons.clearing_fixed_points(**market)[-1]
        induced = np.asarray(anticipated_run(equilibrium, **market)["p"]).ravel()[0]
        assert induced == pytest.approx(equilibrium, abs=0.01)


class TestEachConfigurationReachesItsFixedPoint:
    """The premium and the quality floor between them decide where the price goes."""

    @pytest.mark.parametrize(
        "configuration, reached",
        [
            ("collapse", 0.0),
            ("partial-collapse", 0.6),
            ("no-collapse", 1.25),
        ],
    )
    def test_the_price_settles_where_the_closed_form_says(self, configuration, reached):
        # ``knife-edge`` is excluded deliberately: its map is the identity, so
        # it has no single price to reach. It is covered below.
        market = lemons.MARKETS[configuration]

        # Starting from the top of the quality range, the market lands on the
        # highest price it reproduces.
        assert lemons.clearing_fixed_points(**market)[-1] == pytest.approx(reached)

        path = np.asarray(naive_run(1.0, 40, **market)["p"]).ravel()
        assert path[-1] == pytest.approx(reached, abs=0.02)

    def test_the_knife_edge_stays_where_it_started(self):
        # At a premium of exactly 2 the map's slope is 1, so every price up to
        # the top of the quality range reproduces itself.
        market = lemons.MARKETS["knife-edge"]
        path = np.asarray(naive_run(0.7, 20, **market)["p"]).ravel()
        assert path == pytest.approx(0.7, abs=0.02)

    def test_the_quality_floor_divides_the_two_basins(self):
        # The floor is what makes the collapse partial: a market that starts
        # below it has nobody willing to sell and never recovers, and one that
        # starts above it unravels down to the higher price rather than to zero.
        market = lemons.MARKETS["partial-collapse"]
        floor = market["low"]
        assert np.asarray(naive_run(floor - 0.05, 6, **market)["p"]) == pytest.approx(
            0.0
        )
        above = np.asarray(naive_run(floor + 0.05, 40, **market)["p"]).ravel()
        assert above[-1] == pytest.approx(0.6, abs=0.02)


class TestNoTradeIsAnEquilibriumEverywhere:
    """At ``p = 0`` nobody offers, and that is a price the market reproduces."""

    def test_the_empty_pool_stays_at_zero_rather_than_going_nan(self):
        history = naive_run(0.0, 4)
        assert np.asarray(history["p"]).ravel() == pytest.approx(0.0)
        assert not np.isnan(np.asarray(history["p"])).any()
        assert np.asarray(history["S"]).sum() == 0.0


class TestTheBuyerCanCommitToThePriceInstead:
    """``monopsony_block`` is the same sellers with the price made a decision."""

    @pytest.mark.parametrize("configuration", sorted(lemons.MARKETS))
    def test_the_buyers_surplus_matches_its_closed_form(self, configuration):
        market = lemons.MARKETS[configuration]
        price = lemons.monopsony_price(**market)
        assert buyer_run(price, **market) == pytest.approx(
            lemons.buyer_payoff(price, **market), abs=0.01
        )

    def test_the_monopsonist_pays_less_than_the_competitive_market(self):
        # Both concepts have something to say only where trade survives, and
        # there the buyer's own price is the lower of the two.
        market = lemons.MARKETS["partial-collapse"]
        best = lemons.monopsony_price(**market)
        competitive = lemons.clearing_fixed_points(**market)[-1]
        assert best < competitive

        # And it is a maximum rather than a corner the bounds happened to give.
        assert buyer_run(best, **market) > buyer_run(best - 0.1, **market)
        assert buyer_run(best, **market) > buyer_run(best + 0.1, **market)


class TestAnAcyclicGraphDoesNotMeanASweepSuffices:
    """Three timings, one cyclicity verdict, and three correct treatments."""

    @pytest.mark.parametrize(
        "block",
        [lemons.lemons_block, lemons.naive_lemons_block],
        ids=["anticipated", "naive"],
    )
    def test_the_relevance_graph_sees_one_decision_and_no_cycle(self, block):
        graph = block.relevance_graph(lemons.lemons_calibration())
        assert list(graph.nodes()) == ["S"]
        assert list(graph.edges()) == []
        # So the sweep's own refusal, which fires on a component of more than
        # one decision, would not fire on either: it would solve S against
        # whatever price it was handed and return a rule inconsistent with the
        # price that rule induces.
        assert all(len(component) == 1 for component in graph.condensation())

    def test_only_one_of_the_two_is_correctly_reported_as_isolated(self):
        # The posted-price sellers are paid at a price their own round cannot
        # move, so having no edge is the truth about them.
        calibration = lemons.lemons_calibration()
        assert not lemons.naive_lemons_block.relies_on("S", "S", calibration)
        assert "p" in lemons.naive_lemons_block.get_arrival_states()
        assert "p" not in lemons.lemons_block.get_arrival_states()

    @pytest.mark.xfail(
        strict=True,
        reason="cross-instance reliance is not derivable without expanding the "
        "entity class, so a seller's reliance on the rest of the market is lost",
    )
    def test_a_seller_relies_on_the_other_sellers(self):
        # Where the price is not lagged, a seller's payoff runs through it to
        # every other seller's decision. Reported as an isolated node, the model
        # reads as one decision that can be taken on its own.
        assert lemons.lemons_block.relies_on("S", "S", lemons.lemons_calibration())

    def test_a_committed_price_is_a_second_decision_and_an_order_to_solve_in(self):
        graph = lemons.monopsony_block.relevance_graph(lemons.lemons_calibration())
        assert set(graph.nodes()) == {"p", "S"}
        assert list(graph.edges()) == [("p", "S")]

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
