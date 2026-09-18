"""Tests for skagent.ground (GroundedBlock)."""

import numpy as np
import pytest

import skagent.models.macid as macid
from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock, construct_shocks
from skagent.distributions import Bernoulli, MeanOneLogNormal, Normal
from skagent.ground import Discretized, GroundedBlock, Measure, Sampled

from tests.conftest import RECIPE_CALIBRATION, recipe_block


def instance_block():
    """``recipe_block``'s twin, with the shock declared as a distribution.

    The two blocks describe the same model. They differ only in how ``theta``
    is written down, which is the thing these tests hold against the draws.
    """
    return DBlock(
        **{
            "name": "instance",
            "shocks": {
                "theta": MeanOneLogNormal(sigma=RECIPE_CALIBRATION["sigma_theta"])
            },
            "dynamics": {
                "m": lambda a, theta: a + theta,
                "c": Control(["m"], agent="consumer"),
                "a": lambda m, c: m - c,
                "u": lambda c: c,
            },
            "reward": {"u": "consumer"},
        }
    )


def rng(seed):
    """A generator, since the pair takes one rather than a seed."""
    return np.random.default_rng(seed)


def drawn_sigma(shocks, n=40_000):
    """The standard deviation of the log of ``theta``'s draws."""
    return float(np.std(np.log(shocks["theta"].draw(n))))


class TestShockResolution:
    """The pair resolves the block's declarations; the block is left alone.

    A shock declared as a ``(class, arguments)`` pair is not a distribution
    until a calibration says what its arguments are. Which calibration is the
    grounded block's to decide, so one block may be ground at several.
    """

    def test_shocks_resolve_against_this_instances_calibration(self):
        ground = GroundedBlock(recipe_block(), {"sigma_theta": 0.4})

        assert drawn_sigma(ground.shock_distributions()) == pytest.approx(0.4, abs=0.01)

    def test_one_block_supports_two_calibrations_at_once(self):
        block = recipe_block()
        loose = GroundedBlock(block, {"sigma_theta": 0.4})
        tight = GroundedBlock(block, {"sigma_theta": 0.1})

        loose.shock_distributions()

        assert drawn_sigma(tight.shock_distributions()) == pytest.approx(0.1, abs=0.01)

    def test_shocks_are_resolved_once_so_the_generator_advances(self):
        """Resolving per draw would restart the stream and repeat the draw.

        The distributions hold the generator, so they have to be the same
        distributions from one draw to the next.
        """
        ground = GroundedBlock(
            recipe_block(), dict(RECIPE_CALIBRATION), rng=np.random.default_rng(0)
        )

        first = ground.draw_shocks(5)["theta"]
        second = ground.draw_shocks(5)["theta"]

        assert not np.array_equal(first, second)

    def test_two_instances_over_one_block_draw_their_own_paths(self):
        block = recipe_block()
        seeded = GroundedBlock(
            block, dict(RECIPE_CALIBRATION), rng=np.random.default_rng(7)
        )
        GroundedBlock(block, dict(RECIPE_CALIBRATION), rng=np.random.default_rng(0))

        alone = GroundedBlock(
            recipe_block(), dict(RECIPE_CALIBRATION), rng=np.random.default_rng(7)
        )

        assert np.array_equal(
            seeded.draw_shocks(20)["theta"], alone.draw_shocks(20)["theta"]
        )


class TestTheGeneratorReachesEveryDeclarationStyle:
    """A shock is seeded whether it is declared as a recipe or as an instance.

    ``construct_shocks`` passes the generator to a distribution CONSTRUCTOR, so
    a shock already written as a distribution never sees it. The pair supplies
    it after the fact, which is the only way its ``rng`` argument means the same
    thing for both spellings.
    """

    def test_the_generator_reaches_a_shock_declared_as_an_instance(self):
        block = instance_block()

        one = GroundedBlock(block, {}, rng=np.random.default_rng(1)).draw_shocks(8)
        other = GroundedBlock(block, {}, rng=np.random.default_rng(999)).draw_shocks(8)

        assert not np.array_equal(one["theta"], other["theta"])

    def test_the_declaration_style_does_not_change_the_draws(self):
        """Same model, same seed, same draws -- however the shock is written."""
        as_recipe = GroundedBlock(
            recipe_block(), dict(RECIPE_CALIBRATION), rng=np.random.default_rng(4)
        )
        as_instance = GroundedBlock(instance_block(), {}, rng=np.random.default_rng(4))

        assert np.array_equal(
            as_recipe.draw_shocks(8)["theta"], as_instance.draw_shocks(8)["theta"]
        )

    def test_seeding_leaves_the_blocks_own_distribution_alone(self):
        """The pair seeds its own copies, so a shared block is not re-seeded.

        Blocks are commonly module-level values, and an instance-declared shock
        is an object the block holds rather than a recipe it rebuilds.
        """
        block = instance_block()
        declared = block.shocks["theta"]
        before = declared.rng

        GroundedBlock(block, {}, rng=np.random.default_rng(3)).draw_shocks(8)

        assert block.shocks["theta"] is declared
        assert declared.rng is before


class TestBellmanPeriodIsGrounded:
    """A period is the pair plus a discount factor and a continuation."""

    def test_a_period_is_a_grounded_block(self):
        period = BellmanPeriod(recipe_block(), "beta", dict(RECIPE_CALIBRATION))

        assert isinstance(period, GroundedBlock)

    def test_a_periods_shocks_resolve_against_its_own_calibration(self):
        period = BellmanPeriod(
            recipe_block(), "beta", {"sigma_theta": 0.4, "beta": 0.9}
        )

        assert drawn_sigma(period.shock_distributions()) == pytest.approx(0.4, abs=0.01)


class TestTheCalibrationIsNarrowerThanAScope:
    """What the pair can resolve, and where the wider scope is needed.

    A calibration is settled before a model is solved or simulated. A shock
    argument may instead refer to a value that only exists during a solve or a
    run, and the pair does not hold one, so it is the caller with the value
    that resolves such a shock -- against a scope overlaying it on the
    calibration.
    """

    def volatile_block(self):
        """A block whose shock's spread is a dynamic variable, not a parameter."""
        return DBlock(
            **{
                "name": "volatile",
                "shocks": {"theta": (MeanOneLogNormal, {"sigma": "vol"})},
                "dynamics": {
                    "vol": lambda sigma_theta: sigma_theta,
                    "m": lambda a, theta: a + theta,
                    "c": Control(["m"], agent="consumer"),
                    "a": lambda m, c: m - c,
                    "u": lambda c: c,
                },
                "reward": {"u": "consumer"},
            }
        )

    def test_the_pair_cannot_resolve_a_shock_its_calibration_does_not_cover(self):
        ground = GroundedBlock(self.volatile_block(), {"sigma_theta": 0.4})

        with pytest.raises(KeyError) as raised:
            ground.shock_distributions()

        # The symbol alone would not say which declaration wanted it.
        assert "theta" in str(raised.value)
        assert "sigma=" in str(raised.value)
        assert "vol" in str(raised.value)

    def test_a_scope_overlaying_the_calibration_resolves_it(self):
        block = self.volatile_block()

        resolved = construct_shocks(block.shocks, {"sigma_theta": 0.4, "vol": 0.1})

        assert drawn_sigma(resolved) == pytest.approx(0.1, abs=0.01)


class TestRepointingTheGenerator:
    """``with_rng`` gives a new pair rather than repointing this one.

    A holder drawing from a pair must not have its path changed underneath it
    by someone else asking for a different generator, so the operation returns
    a copy. What changes is the sample; the block and the calibration -- the
    model -- do not.
    """

    def test_the_original_keeps_its_own_path(self):
        ground = GroundedBlock(recipe_block(), dict(RECIPE_CALIBRATION), rng=rng(0))
        before = ground.draw_shocks(3)["theta"]

        ground.with_rng(rng(999))

        again = GroundedBlock(
            recipe_block(), dict(RECIPE_CALIBRATION), rng=rng(0)
        ).draw_shocks(3)["theta"]
        assert np.array_equal(before, again)

    def test_the_copy_draws_the_path_its_generator_asks_for(self):
        ground = GroundedBlock(recipe_block(), dict(RECIPE_CALIBRATION), rng=rng(0))

        copied = ground.with_rng(rng(4))
        direct = GroundedBlock(recipe_block(), dict(RECIPE_CALIBRATION), rng=rng(4))

        assert np.array_equal(
            copied.draw_shocks(3)["theta"], direct.draw_shocks(3)["theta"]
        )

    def test_the_two_share_no_distribution(self):
        ground = GroundedBlock(recipe_block(), dict(RECIPE_CALIBRATION), rng=rng(0))
        ground.shock_distributions()

        copied = ground.with_rng(rng(1))

        assert (
            copied.shock_distributions()["theta"]
            is not (ground.shock_distributions()["theta"])
        )

    def test_a_period_is_copied_as_a_period(self):
        period = BellmanPeriod(
            recipe_block(), "beta", dict(RECIPE_CALIBRATION), rng=rng(0)
        )

        copied = period.with_rng(rng(1))

        # Whatever a subclass adds is carried over, so the copy is usable as
        # the thing it was copied from.
        assert isinstance(copied, BellmanPeriod)
        assert copied.discount_variable == period.discount_variable
        assert copied.arrival_states == period.arrival_states


def linear_block():
    """A payoff linear in one normal shock, so its expectation is closed-form.

    ``a`` arrives, ``theta`` is drawn, and the consumer's payoff is whatever
    its rule returns. Under a rule that spends a fixed share of ``m``, the
    expected payoff is that share of ``a + E[theta]`` exactly -- which is what
    lets a reduction be checked against arithmetic rather than against another
    reduction.
    """
    return DBlock(
        **{
            "name": "linear",
            "shocks": {"theta": (Normal, {"mu": "mu", "sigma": "sigma"})},
            "dynamics": {
                "m": lambda a, theta: a + theta,
                "c": Control(["m"], agent="consumer"),
                "u": lambda c: c,
            },
            "reward": {"u": "consumer"},
        }
    )


LINEAR_CALIBRATION = {"mu": 1.0, "sigma": 2.0, "a": 3.0}
HALF = {"c": lambda m: m / 2}


def linear_ground(**calibration):
    return GroundedBlock(linear_block(), dict(LINEAR_CALIBRATION, **calibration))


class TestWhatAProfileIsWorth:
    """``expected_payoff`` under each of the two reductions."""

    def test_the_discretization_is_exact_on_a_linear_payoff(self):
        # A discretization is a rule of nodes and weights, and the Gauss-Hermite
        # rule behind a normal shock integrates a linear integrand exactly. So
        # this is an equality against the closed form rather than a tolerance,
        # and three nodes reach it as well as the default seven do.
        ground = linear_ground()
        closed_form = (LINEAR_CALIBRATION["a"] + LINEAR_CALIBRATION["mu"]) / 2

        assert float(ground.expected_payoff(HALF, Discretized())) == pytest.approx(
            closed_form, abs=1e-12
        )
        assert float(
            ground.expected_payoff(HALF, Discretized({"theta": {"n_points": 3}}))
        ) == pytest.approx(closed_form, abs=1e-12)

    def test_a_sampled_estimate_reaches_the_same_number(self):
        # The other reduction answers the same question with sampling error in
        # place of a rule, so it agrees to within its own standard error.
        ground = linear_ground()
        closed_form = (LINEAR_CALIBRATION["a"] + LINEAR_CALIBRATION["mu"]) / 2

        estimate = ground.expected_payoff(HALF, Sampled(40_000, rng=rng(0)))

        assert float(estimate) == pytest.approx(closed_form, abs=0.05)

    def test_a_torch_space_rule_can_be_scored_once_it_is_wrapped(self):
        # What NUMPYRULE is for: a policy built in torch -- a trained network,
        # here a closed form standing in for one -- is worth the same as the
        # numpy rule it agrees with, and could not be handed to this machinery
        # at all before, since a torch rule stacks its arguments.
        #
        # The agreement is to float32 and not to the 1e-12 the discretization
        # reaches on its own: the wrapper converts at the default dtype the
        # networks are trained in, so the boundary costs precision even where
        # the quadrature is exact. Asking for float64 recovers it, which is why
        # the dtype is an argument.
        import torch

        from skagent.algos.vfi import numpy_decision_rule

        def torch_half(m):
            return torch.as_tensor(m) / 2

        ground = linear_ground()
        exact = float(ground.expected_payoff(HALF, Discretized()))

        single = {"c": numpy_decision_rule(torch_half)}
        assert float(ground.expected_payoff(single, Discretized())) == pytest.approx(
            exact, abs=1e-6
        )

        double = {"c": numpy_decision_rule(torch_half, dtype=torch.float64)}
        assert float(ground.expected_payoff(double, Discretized())) == pytest.approx(
            exact, abs=1e-12
        )

    def test_two_profiles_measured_on_the_same_draws_differ_only_by_the_profile(self):
        # Comparing two profiles is the reason to ask what one is worth, and
        # under independent draws the comparison carries both estimates' noise.
        # Measured against generators in the same state it carries neither: the
        # payoff here is proportional to the share spent, so the ratio is the
        # ratio of the shares EXACTLY, which it would not be otherwise.
        ground = linear_ground()
        third = {"c": lambda m: m / 3}

        half_worth = ground.expected_payoff(HALF, Sampled(2_000, rng=rng(7)))
        third_worth = ground.expected_payoff(third, Sampled(2_000, rng=rng(7)))

        assert float(half_worth) / float(third_worth) == pytest.approx(1.5, abs=1e-12)

    def test_the_answer_says_which_axis_it_reduced(self):
        # A number that has lost its measure cannot be held to a tolerance, so
        # the axis and its size travel with it. The node count is the size of
        # the integration grid, which is the product over the shocks and is not
        # something the caller stated.
        ground = linear_ground()

        sampled = ground.expected_payoff(HALF, Sampled(64, rng=rng(0)))
        assert (sampled.axis, sampled.size) == ("samples", 64)

        discretized = ground.expected_payoff(
            HALF, Discretized({"theta": {"n_points": 5}})
        )
        assert (discretized.axis, discretized.size) == ("nodes", 5)

    def test_a_discrete_shock_is_taken_as_the_points_it_already_has(self):
        # Bernoulli needs no rule: its own two points ARE the integration
        # nodes, so the expectation is the exact weighted payoff.
        block = DBlock(
            **{
                "name": "coin",
                "shocks": {"heads": (Bernoulli, {"p": "p"})},
                "dynamics": {"u": lambda heads: 10.0 * heads},
                "reward": {"u": "better"},
            }
        )
        ground = GroundedBlock(block, {"p": 0.3})

        worth = ground.expected_payoff({}, Discretized())

        assert float(worth) == pytest.approx(3.0, abs=1e-12)
        assert (worth.axis, worth.size) == ("nodes", 2)

    def test_a_block_already_discretized_is_integrated_over_its_own_nodes(self):
        # ``Block.discretize`` returns a block whose shocks are the nodes, and
        # a pair over one of those has nothing left to discretize. Reducing it
        # must reach the same number as asking the original for the same rule.
        nodes = {"theta": {"n_points": 5}}
        pre_discretized = GroundedBlock(
            linear_block().discretize(nodes, calibration=LINEAR_CALIBRATION),
            LINEAR_CALIBRATION,
        )

        worth = pre_discretized.expected_payoff(HALF, Discretized())

        assert float(worth) == pytest.approx(
            float(linear_ground().expected_payoff(HALF, Discretized(nodes))),
            abs=1e-12,
        )
        assert (worth.axis, worth.size) == ("nodes", 5)

    def test_a_block_with_no_shocks_is_worth_what_it_pays(self):
        # A static profile with nothing random in it has no axis to reduce, and
        # its payoff is already its own expectation.
        ground = GroundedBlock(macid.prisoners_dilemma_block, {})
        defect_against_cooperate = {"D1": lambda: 1.0, "D2": lambda: 0.0}

        worth = ground.expected_payoff(
            defect_against_cooperate, Discretized(), agent="player_1"
        )

        assert float(worth) == 5.0
        assert worth.size == 1

    def test_the_payoff_is_the_one_agent_s_and_not_the_table_s(self):
        # Two agents' rewards summed is nobody's objective, so an agent is how
        # a multi-agent block is asked.
        ground = GroundedBlock(macid.prisoners_dilemma_block, {})
        mutual_defection = {"D1": lambda: 1.0, "D2": lambda: 1.0}

        for agent in ("player_1", "player_2"):
            worth = ground.expected_payoff(mutual_defection, Discretized(), agent=agent)
            assert float(worth) == 1.0

    def test_an_agent_that_owns_no_reward_is_refused(self):
        # Its payoff would otherwise be an empty sum, which is zero and reads
        # as an answer.
        ground = GroundedBlock(macid.prisoners_dilemma_block, {})

        with pytest.raises(ValueError, match="no reward in this block"):
            ground.expected_payoff(
                {"D1": lambda: 1.0, "D2": lambda: 1.0}, Sampled(4), agent="player_3"
            )

    def test_the_arrival_states_are_the_ones_given(self):
        # ``a`` arrives rather than being calibrated, and the profile is worth
        # something different at each value of it.
        ground = GroundedBlock(linear_block(), {"mu": 1.0, "sigma": 2.0})

        worth = ground.expected_payoff(HALF, Discretized(), states={"a": 9.0})

        assert float(worth) == pytest.approx(5.0, abs=1e-12)

    def test_a_measure_is_one_type_with_one_contract(self):
        # The two reductions are interchangeable at the call site and differ
        # only in what they are configured with, which is what the shared type
        # says; anything else answering to it serves.
        assert issubclass(Sampled, Measure) and issubclass(Discretized, Measure)

        with pytest.raises(TypeError, match="abstract"):
            Measure()

    @pytest.mark.parametrize("n", [0, -1, 2.5, True])
    def test_a_sample_count_is_a_positive_integer(self, n):
        with pytest.raises(ValueError, match="positive integer"):
            Sampled(n)
