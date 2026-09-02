"""Entity declarations on blocks, and what the simulator does with them."""

import numpy as np
import pytest

import skagent.models.cournot as cournot
from skagent.block import Aggregate, Control, DBlock, Entity, RBlock
from skagent.distributions import Normal, Uniform
from skagent.simulation.monte_carlo import Simulator

A, B = 10.0, 1.0


def cournot_block():
    """The smallest model with an entity, an agent role and a crossing.

    Firms draw a cost and choose a quantity; the market clears on the average
    quantity; the firms are paid. The entity is declared twice because the
    population acts before the market clears and is paid after, which is also
    what fixes the dynamics order.
    """
    return RBlock(
        name="cournot",
        blocks=[
            RBlock(
                name="firms",
                entity=Entity("firm"),
                blocks=[
                    DBlock(
                        name="offer",
                        shocks={"c": (Uniform, {"low": "cl", "high": "ch"})},
                        dynamics={"q": Control(["c"], agent="firm")},
                    )
                ],
            ),
            DBlock(
                name="market",
                dynamics={
                    "Q": lambda q: q.mean(),
                    "P": lambda A, b, Q: A - b * Q,
                },
            ),
            RBlock(
                name="payoffs",
                entity=Entity("firm"),
                blocks=[
                    DBlock(
                        name="payoff",
                        dynamics={"u": lambda P, c, q: (P - c) * q},
                        reward={"u": "firm"},
                    )
                ],
            ),
        ],
    )


def collusion_calibration(size=3):
    return {"A": A, "b": B, "cl": 4.0, "ch": 4.0, "firm": size}


def fixed_rules(quantities):
    """One decision rule per firm, so a profile may be asymmetric."""
    return np.array([(lambda v: lambda c: v)(v) for v in quantities])


class TestTheDeclaration:
    """What a block reports about the entity classes it declares."""

    def test_a_variable_belongs_to_the_entity_of_the_block_defining_it(self):
        signatures = cournot_block().signatures()

        assert signatures["c"] == frozenset({"firm"})
        assert signatures["q"] == frozenset({"firm"})
        assert signatures["u"] == frozenset({"firm"})
        assert signatures["Q"] == frozenset()
        assert signatures["P"] == frozenset()

    def test_reading_out_of_an_entity_is_reported_as_a_crossing(self):
        crossings = cournot_block().crossings()

        assert crossings == {"Q": [("q", frozenset({"firm"}), frozenset())]}

    def test_reading_into_an_entity_is_a_broadcast_and_is_not_reported(self):
        # ``u`` is per firm and reads the single price. Nothing has to be
        # decided about that, so it is not a crossing.
        assert "u" not in cournot_block().crossings()

    def test_declaration_order_fixes_the_dynamics_order(self):
        assert list(cournot_block().get_dynamics()) == ["q", "Q", "P", "u"]

    def test_an_agent_role_is_attributed_to_its_blocks_entity(self):
        assert cournot_block().agent_populations() == {"firm": "firm"}


class TestARoleIsNotAnEntity:
    """An agent name declares a role; it never creates an entity class."""

    def study_block(self):
        return RBlock(
            name="study",
            blocks=[
                RBlock(
                    name="subjects",
                    entity=Entity("subject"),
                    blocks=[
                        DBlock(
                            name="report",
                            dynamics={"d": Control(["b"], agent="subject")},
                        )
                    ],
                ),
                DBlock(
                    name="analysis",
                    dynamics={"f": Control(["d"], agent="analyst")},
                ),
            ],
        )

    def test_a_role_in_a_block_with_no_entity_has_no_population(self):
        # There is exactly one analyst. Reporting it as a population of one
        # would make a modelling claim the block does not make.
        assert self.study_block().agent_populations() == {
            "subject": "subject",
            "analyst": None,
        }

    def test_an_estimator_over_a_population_is_a_crossing(self):
        assert self.study_block().crossings() == {
            "f": [("d", frozenset({"subject"}), frozenset())]
        }


class TestGetControls:
    """A composed block reports its controls the way a leaf block does."""

    def test_an_rblock_returns_a_mapping(self):
        """`RBlock` overrode this with a list, so `.items()` failed on it."""
        controls = cournot.cournot_block.get_controls()

        assert {sym: control.agent for sym, control in controls.items()} == {
            "q": "firm"
        }

    def test_a_leaf_and_a_composed_block_agree_in_type(self):
        leaf = DBlock(
            name="leaf", dynamics={"a": Control([], agent="p"), "u": lambda a: -a}
        )
        composed = RBlock(name="composed", blocks=[leaf])

        assert type(composed.get_controls()) is type(leaf.get_controls())


class TestGetControl:
    """The block is what knows which of its symbols are controls."""

    def block(self):
        return DBlock(
            name="one_player",
            dynamics={"a": Control([]), "u": lambda a: -a},
            reward={"u": "p"},
        )

    def test_a_control_is_returned(self):
        assert self.block().get_control("a").iset == []

    def test_a_symbol_that_is_not_a_control_raises(self):
        """A reward variable is a dynamic, but it is not a decision."""
        with pytest.raises(ValueError, match="not a control"):
            self.block().get_control("u")

    def test_an_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="not a control"):
            self.block().get_control("nonesuch")


class TestTheDecidingAgent:
    """Whose payoff a control maximizes, read off the block."""

    def test_each_control_reports_its_own_agent(self):
        block = DBlock(
            name="two_players",
            dynamics={
                "a1": Control([], agent="p1"),
                "a2": Control([], agent="p2"),
                "u1": lambda a1, a2: a1 - a2,
                "u2": lambda a1, a2: a2 - a1,
            },
            reward={"u1": "p1", "u2": "p2"},
        )

        assert block.deciding_agent("a1") == "p1"
        assert block.deciding_agent("a2") == "p2"

    def test_one_owner_needs_no_attribution(self):
        """With a single owner there is nothing to disambiguate."""
        block = DBlock(
            name="one_player",
            dynamics={"a": Control([]), "u": lambda a: -a},
            reward={"u": "p"},
        )

        assert block.deciding_agent("a") is None


class TestAnAggregateEscapesItsEntity:
    """An economy-wide shock may sit beside the equations that read it."""

    def test_an_aggregate_shock_is_axis_free_where_it_is_declared(self):
        block = RBlock(
            name="economy",
            entity=Entity("household"),
            blocks=[
                DBlock(
                    name="inner",
                    shocks={
                        "tfp": Aggregate(Normal(1.0, 0.1)),
                        "eps": Normal(0.0, 1.0),
                    },
                    dynamics={"y": lambda tfp, eps: tfp + eps},
                )
            ],
        )

        signatures = block.signatures()
        assert signatures["tfp"] == frozenset()
        assert signatures["eps"] == frozenset({"household"})


class TestCardinalityComesFromCalibration:
    """An entity has a name and no size, so the count is read per run."""

    def test_a_declared_entity_needs_a_cardinality(self):
        calibration = collusion_calibration()
        del calibration["firm"]

        with pytest.raises(ValueError, match="no key 'firm'"):
            Simulator(calibration, cournot_block(), {"q": lambda c: 1.0}, {})

    def test_a_cardinality_must_be_a_positive_integer(self):
        with pytest.raises(ValueError, match="at least one instance"):
            Simulator(
                collusion_calibration(size=0),
                cournot_block(),
                {"q": lambda c: 1.0},
                {},
            )


class TestHistoryShape:
    """History carries the period axis, the sample axis, then the entity axes."""

    def simulate(self, size=3, sample_count=4):
        sim = Simulator(
            collusion_calibration(size),
            cournot_block(),
            {"q": lambda c: 1.0},
            {},
            sample_count=sample_count,
            T_sim=5,
            seed=0,
        )
        sim.initialize_sim()
        return sim.simulate()

    def test_an_attribute_carries_its_entitys_axis(self):
        history = self.simulate()

        assert history["c"].shape == (5, 4, 3)
        assert history["q"].shape == (5, 4, 3)
        assert history["u"].shape == (5, 4, 3)

    def test_an_axis_free_variable_does_not(self):
        history = self.simulate()

        assert history["Q"].shape == (5, 4)
        assert history["P"].shape == (5, 4)

    def test_a_block_with_no_entity_keeps_the_shape_it_always_had(self):
        block = DBlock(
            name="plain",
            shocks={"eps": Normal(0.0, 1.0)},
            dynamics={"x": lambda eps: eps},
        )
        sim = Simulator({}, block, {}, {}, sample_count=7, T_sim=3, seed=0)
        sim.initialize_sim()
        history = sim.simulate()

        assert history["eps"].shape == (3, 7)
        assert history["x"].shape == (3, 7)


class TestInstancesAreHeterogeneous:
    """A per-instance shock is drawn once per instance, not once per sample."""

    def test_each_firm_draws_its_own_cost(self):
        sim = Simulator(
            {"A": A, "b": B, "cl": 2.0, "ch": 6.0, "firm": 5},
            cournot_block(),
            {"q": lambda c: c},
            {},
            sample_count=2,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        costs = sim.simulate()["c"][0]

        # Five distinct costs per sample, and the two samples differ.
        assert len(np.unique(costs[0])) == 5
        assert not np.array_equal(costs[0], costs[1])


class TestTheAggregationIsOverInstancesOnly:
    """A reduction must not average over the sample axis as well."""

    def test_the_average_is_taken_within_each_sample(self):
        sim = Simulator(
            {"A": A, "b": B, "cl": 2.0, "ch": 6.0, "firm": 4},
            cournot_block(),
            {"q": lambda c: c},
            {},
            sample_count=6,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        history = sim.simulate()

        # Each sample is its own market, so each has its own average quantity.
        assert history["Q"].shape == (1, 6)
        assert np.allclose(history["Q"][0], history["q"][0].mean(axis=-1))
        assert len(np.unique(history["Q"][0])) > 1


class TestShapeValidation:
    """An equation returning the wrong shape is caught where it happens."""

    def test_an_axis_free_equation_returning_an_array_raises(self):
        # ``Q`` is declared outside every entity but forgets to reduce.
        block = RBlock(
            name="broken",
            blocks=[
                RBlock(
                    name="firms",
                    entity=Entity("firm"),
                    blocks=[
                        DBlock(
                            name="offer",
                            shocks={"c": (Uniform, {"low": "cl", "high": "ch"})},
                            dynamics={"q": Control(["c"], agent="firm")},
                        )
                    ],
                ),
                DBlock(name="market", dynamics={"Q": lambda q: q * 2}),
            ],
        )
        sim = Simulator(
            collusion_calibration(),
            block,
            {"q": lambda c: 1.0},
            {},
            sample_count=1,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()

        with pytest.raises(ValueError, match="outside every entity"):
            sim.simulate()

    def test_a_non_finite_value_entering_a_reduction_raises(self):
        # One firm's bad value would otherwise become every firm's price.
        sim = Simulator(
            collusion_calibration(),
            cournot_block(),
            {"q": lambda c: np.where(np.arange(len(c)) == 0, np.nan, 1.0)},
            {},
            sample_count=1,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()

        with pytest.raises(ValueError, match="1 non-finite entry"):
            sim.simulate()


class TestCournotArithmetic:
    """The model's own analytic claims, which shapes alone would not catch."""

    def test_expected_profit_carries_the_variance_of_costs(self):
        # An aggregate-only claim survives a population that has lost its
        # cross-section, because a mean does. E[u] carries the second moment.
        m, v = 4.0, (6.0 - 2.0) ** 2 / 12
        sim = Simulator(
            {"A": A, "b": B, "cl": 2.0, "ch": 6.0, "firm": 200},
            cournot_block(),
            {"q": lambda c: (A - c) / (2 * B)},
            {},
            sample_count=400,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        history = sim.simulate()

        assert history["Q"].mean() == pytest.approx((A - m) / (2 * B), rel=0.01)
        assert history["P"].mean() == pytest.approx((A + m) / 2, rel=0.01)
        assert history["u"].mean() == pytest.approx(
            ((A - m) ** 2 / 2 + v) / (2 * B), rel=0.01
        )

    @pytest.mark.parametrize(
        "quantities,price,profits",
        [
            ([4.5, 4.5, 4.5], 5.5, [6.75, 6.75, 6.75]),
            ([3.0, 3.0, 3.0], 7.0, [9.0, 9.0, 9.0]),
            ([6.0, 3.0, 3.0], 6.0, [12.0, 6.0, 6.0]),
        ],
        ids=["cournot-nash", "joint-monopoly", "one-defects"],
    )
    def test_the_three_collusion_profiles(self, quantities, price, profits):
        # The defection profile is an asymmetric ASSIGNMENT of rules over a
        # symmetric model: three firms, three rules, two of them equal. The
        # model does not change, and no rule reads a firm's position.
        sim = Simulator(
            collusion_calibration(),
            cournot_block(),
            {"q": fixed_rules(quantities)},
            {},
            sample_count=1,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        history = sim.simulate()

        assert history["q"][0, 0] == pytest.approx(quantities)
        assert history["P"][0, 0] == pytest.approx(price)
        assert history["u"][0, 0] == pytest.approx(profits)


class TestDegenerateDistributionsDraw:
    """A zero-spread distribution is a point mass, not a NaN.

    The collusion configuration needs cost degenerate at one value, which is a
    uniform whose bounds coincide. Drawing by inverting the quantile function
    returns NaN there, though sampling directly does not.
    """

    def test_a_uniform_with_coincident_bounds_is_a_point_mass(self):
        drawn = Uniform(4.0, 4.0, rng=np.random.default_rng(0)).draw(3)

        assert np.array_equal(drawn, np.full(3, 4.0))

    def test_a_normal_with_zero_spread_is_a_point_mass(self):
        drawn = Normal(1.0, 0.0, rng=np.random.default_rng(0)).draw(3)

        assert np.array_equal(drawn, np.full(3, 1.0))


class TestWhatDoesNotWorkYet:
    """The parts of the entity feature that are declared but not honoured.

    Each of these is a wrong answer rather than a missing feature: the query
    succeeds and returns something a reader would act on. Marked strict, so that
    implementing any of them fails here and the marker comes off.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="the relevance graph is not computed on an entity expansion, so "
        "it reports one node with no edges",
    )
    def test_a_firm_relies_on_the_other_firms(self):
        block = cournot_block()
        calibration = collusion_calibration()

        # Each firm's payoff runs through the average quantity to the price, so
        # every firm relies on every other. Presented as a single node with no
        # edges, the model reads as one decision taken in isolation -- an
        # acyclic singleton, which is also what the cyclicity test looks for.
        assert block.relies_on("q", "q", calibration)

    @pytest.mark.xfail(
        strict=True,
        reason="the relevance graph is not entity-attributed, so cross-instance "
        "reliance has nowhere to appear",
    )
    def test_the_relevance_graph_shows_the_cross_instance_edge(self):
        graph = cournot_block().relevance_graph(collusion_calibration())

        # A class relying on itself is one node with a self-loop, which is what
        # plate notation draws and what distinguishes this from a lone decision.
        assert graph.edges()

    @pytest.mark.xfail(
        strict=True,
        reason="ModelAnalyzer infers populations from agent attribution rather "
        "than reading the block's entity metadata",
    )
    def test_the_analyzer_reads_the_declared_entity(self):
        from skagent.model_analyzer import ModelAnalyzer

        analyzer = ModelAnalyzer(cournot_block(), collusion_calibration())

        # Without this the model diagram draws Cournot as though it had no
        # population and no aggregation.
        assert "firm" in analyzer.plates

    @pytest.mark.xfail(
        strict=True,
        reason="the solvers do not yet refuse a block with a crossing",
    )
    def test_a_bellman_period_refuses_a_block_with_a_crossing(self):
        from skagent.bellman import BellmanPeriod

        calibration = dict(collusion_calibration(), beta=0.9)

        # A solver optimises at each grid point independently, which is not what
        # an aggregate is. Accepting the block means answering a question the
        # model does not pose.
        with pytest.raises(ValueError):
            BellmanPeriod(cournot_block(), "beta", calibration)


class TestTheShippedModel:
    """``skagent.models.cournot`` agrees with the claims in its own docstring."""

    def test_the_closed_forms_are_the_documented_quantities(self):
        assert cournot.nash_quantity() == pytest.approx(4.5)
        assert cournot.monopoly_quantity() == pytest.approx(3.0)

    def test_the_heterogeneous_market_converges_to_its_analytic_moments(self):
        want = cournot.analytic_moments()
        sim = Simulator(
            cournot.heterogeneous_calibration(),
            cournot.cournot_block,
            {"q": cournot.competitive_rule},
            {},
            sample_count=400,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        history = sim.simulate()

        assert history["Q"].mean() == pytest.approx(want["Q"], rel=0.01)
        assert history["P"].mean() == pytest.approx(want["P"], rel=0.01)
        assert history["u"].mean() == pytest.approx(want["E[u]"], rel=0.01)

    @pytest.mark.parametrize("profile", list(cournot.PROFILES))
    def test_each_shipped_profile_pays_what_it_claims(self, profile):
        totals = {"cournot-nash": 20.25, "joint-monopoly": 27.0, "one-defects": 24.0}
        sim = Simulator(
            cournot.collusion_calibration(),
            cournot.cournot_block,
            {"q": cournot.profile_rules(cournot.PROFILES[profile])},
            {},
            sample_count=1,
            T_sim=1,
            seed=0,
        )
        sim.initialize_sim()
        history = sim.simulate()

        assert history["q"][0, 0] == pytest.approx(cournot.PROFILES[profile])
        assert history["u"][0, 0].sum() == pytest.approx(totals[profile])

    def test_collusion_beats_nash_and_defection_beats_collusion(self):
        # The dilemma itself, rather than three unrelated numbers.
        def total_and_first(profile):
            sim = Simulator(
                cournot.collusion_calibration(),
                cournot.cournot_block,
                {"q": cournot.profile_rules(cournot.PROFILES[profile])},
                {},
                sample_count=1,
                T_sim=1,
                seed=0,
            )
            sim.initialize_sim()
            u = sim.simulate()["u"][0, 0]
            return u.sum(), u[0]

        nash_total, _ = total_and_first("cournot-nash")
        collude_total, collude_each = total_and_first("joint-monopoly")
        defect_total, defector = total_and_first("one-defects")

        assert collude_total > nash_total
        assert defector > collude_each
        assert defect_total < collude_total
