"""The lemons market: adverse selection, and a fixed point an acyclic graph hides."""

import numpy as np
import pytest
import torch

import skagent.models.lemons as lemons
from skagent.algos.tabular import TabularBestResponseSolver
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator

SIZE = 50000
TOP = lemons.QUALITY_HIGH


def play(block, calibration, rules, arrival=None, periods=1, seed=0):
    sim = Simulator(
        calibration,
        block,
        rules,
        {} if arrival is None else arrival,
        sample_count=1,
        T_sim=periods,
        seed=seed,
    )
    sim.initialize_sim()
    return sim.simulate()


def posted_run(p0, periods, **market):
    """Simulate the posted-price market from *p0*; each period is one round."""
    return play(
        lemons.naive_lemons_block,
        lemons.lemons_calibration(size=SIZE, **market),
        {"S": lemons.seller_rule},
        {"p": p0},
        periods,
    )


def anticipated_run(price, **market):
    """Play the anticipated-price market once, with sellers expecting *price*."""
    return play(
        lemons.lemons_block,
        lemons.lemons_calibration(size=SIZE, **market),
        {"S": lemons.supply_rule(price)},
    )


def peaches_run(p0, periods, **market):
    """Simulate the posted-price market for peaches and lemons."""
    return play(
        lemons.naive_peaches_block,
        lemons.peaches_calibration(size=SIZE, **market),
        {"S": lemons.seller_rule},
        {"p": p0},
        periods,
    )


def buyer_run(price, **market):
    """Play the monopsony market once at *price*, and return the buyer's surplus."""
    history = play(
        lemons.monopsony_block,
        lemons.lemons_calibration(size=SIZE, **market),
        {"p": lemons.bid_rule(price), "S": lemons.seller_rule},
    )
    return np.asarray(history["w"]).ravel()[0]


class TestTheClearingPriceRunsOnEveryPath:
    """It is a weighted mean, so no path has to special-case it."""

    def test_it_agrees_with_a_mean_over_the_items_that_sold(self):
        theta = np.random.default_rng(0).uniform(0, TOP, 5000)
        for price in (0.6, 1.5, TOP):
            S = lemons.seller_rule(theta, price)
            assert lemons.clearing_price(theta, S, 1.5) == pytest.approx(
                1.5 * theta[S > 0.5].mean()
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
        path = np.asarray(posted_run(TOP, 10)["p"]).ravel()
        assert path == pytest.approx(lemons.clearing_path(TOP, 10), abs=0.02)

    def test_the_market_trades_before_it_collapses(self):
        # Without this the collapse below is indistinguishable from a market
        # that never traded. The payoff block sits before the market block, so
        # a seller is paid at the price it responded to and never overpays.
        history = posted_run(TOP, 10)
        volume = np.asarray(history["S"]).sum(axis=-1).ravel()
        assert volume[0] > 0.9 * SIZE
        assert volume[-1] < 0.1 * SIZE
        assert (np.asarray(history["u"]) >= 0).all()


class TestAnticipationMakesItAFixedPointInRules:
    """Nothing is lagged, so the equilibrium has to be solved for rather than run."""

    @pytest.mark.parametrize("anticipated", [2.2, 1.0, 0.3])
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
        assert induced == pytest.approx(equilibrium, abs=0.02)


class TestEachUniformConfigurationReachesItsFixedPoint:
    """The premium and the quality floor between them decide where the price goes."""

    @pytest.mark.parametrize(
        "configuration, reached",
        [("akerlof", 0.0), ("partial-collapse", 1.2), ("no-collapse", 2.5)],
    )
    def test_the_price_settles_where_the_closed_form_says(self, configuration, reached):
        # ``knife-edge`` is excluded deliberately: its map is the identity, so
        # it has no single price to reach. It is covered below.
        market = lemons.MARKETS[configuration]

        # Starting from the top of the quality range, the market lands on the
        # highest price it reproduces.
        assert lemons.clearing_fixed_points(**market)[-1] == pytest.approx(reached)

        path = np.asarray(posted_run(TOP, 40, **market)["p"]).ravel()
        assert path[-1] == pytest.approx(reached, abs=0.03)

    def test_the_knife_edge_stays_where_it_started(self):
        # At a premium of exactly 2 the map's slope is 1, so every price up to
        # the top of the quality range reproduces itself.
        path = np.asarray(posted_run(1.4, 20, **lemons.MARKETS["knife-edge"])["p"])
        assert path.ravel() == pytest.approx(1.4, abs=0.03)

    def test_the_quality_floor_divides_the_two_basins(self):
        # The floor is what makes the collapse partial: a market that starts
        # below it has nobody willing to sell and never recovers, and one that
        # starts above it unravels down to the higher price rather than to zero.
        market = lemons.MARKETS["partial-collapse"]
        floor = market["low"]
        assert np.asarray(posted_run(floor - 0.05, 6, **market)["p"]) == pytest.approx(
            0.0
        )
        above = np.asarray(posted_run(floor + 0.05, 40, **market)["p"]).ravel()
        assert above[-1] == pytest.approx(1.2, abs=0.03)


class TestTwoTypesGiveAPriceTheUniformMarketCannot:
    """Akerlof's cars: a peach or a lemon, and a market that need not collapse."""

    @pytest.mark.parametrize("p0, reached", [(0.3, 0.0), (1.0, 0.6), (2.5, 2.28)])
    def test_where_the_market_lands_depends_on_where_it_starts(self, p0, reached):
        market = lemons.PEACH_MARKETS["two-prices"]
        assert any(
            reached == pytest.approx(price)
            for price in lemons.peaches_fixed_points(**market)
        )

        path = np.asarray(peaches_run(p0, 25, **market)["p"]).ravel()
        assert path[-1] == pytest.approx(reached, abs=0.03)

    def test_the_middle_price_is_a_market_in_lemons_alone(self):
        # This is what the uniform market has no room for: trade survives, and
        # every peach is withheld. The price is a lemon's worth times the
        # premium and does not depend on what a peach is worth or how many
        # there are.
        market = lemons.PEACH_MARKETS["two-prices"]
        history = peaches_run(1.0, 25, **market)
        sold = np.asarray(history["S"])[-1].ravel()
        quality = np.asarray(history["theta"])[-1].ravel()

        assert sold.mean() == pytest.approx(1 - market["share"], abs=0.02)
        assert quality[sold > 0.5].max() == pytest.approx(0.4)

    def test_peaches_trade_only_where_they_are_common_enough(self):
        # Below the critical share no price a buyer will pay for the average car
        # is enough to bring a peach out, however the market is started.
        critical = lemons.peach_share_for_trade()
        assert critical == pytest.approx(0.5833, abs=1e-4)

        assert len(lemons.peaches_fixed_points(share=critical - 0.05)) == 2
        assert len(lemons.peaches_fixed_points(share=critical + 0.05)) == 3

        scarce = lemons.PEACH_MARKETS["lemons-only"]
        assert scarce["share"] < critical
        path = np.asarray(peaches_run(2.5, 25, **scarce)["p"]).ravel()
        assert path[-1] == pytest.approx(0.6, abs=0.03)


class TestNoTradeIsAnEquilibriumEverywhere:
    """At ``p = 0`` nobody offers, and that is a price the market reproduces."""

    def test_the_empty_pool_stays_at_zero_rather_than_going_nan(self):
        history = posted_run(0.0, 4)
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
        assert best < lemons.clearing_fixed_points(**market)[-1]

        # And it is a maximum rather than a corner the bounds happened to give.
        assert buyer_run(best, **market) > buyer_run(best - 0.6, **market)
        assert buyer_run(best, **market) > buyer_run(best + 0.6, **market)


class TestThreeTimingsAndThreeTreatments:
    """One decision, three timings, and a different verdict on each."""

    def test_an_anticipated_price_makes_a_seller_rely_on_the_others(self):
        # A seller's payoff runs through the price it anticipates to every other
        # seller's decision, so the model is a strategic fixed point among the
        # instances of one class.
        graph = lemons.lemons_block.relevance_graph(lemons.lemons_calibration())
        assert list(graph.nodes()) == ["S"] and list(graph.edges()) == [("S", "S")]
        assert not graph.is_acyclic()

        # And the count of a component's members does not see it: there is one
        # decision to solve, and it is its own predecessor. A schedule that read
        # cyclicity off that count would solve S against whatever price it was
        # handed and return a rule inconsistent with the price that rule induces.
        assert all(len(component) == 1 for component in graph.condensation())

    def test_a_posted_price_leaves_each_seller_on_its_own(self):
        # The posted-price sellers are paid at a price their own round cannot
        # move, so having no edge is the truth about them.
        calibration = lemons.lemons_calibration()
        graph = lemons.naive_lemons_block.relevance_graph(calibration)
        assert list(graph.nodes()) == ["S"] and list(graph.edges()) == []
        assert not lemons.naive_lemons_block.relies_on("S", "S", calibration)
        assert "p" in lemons.naive_lemons_block.get_arrival_states()
        assert "p" not in lemons.lemons_block.get_arrival_states()

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
