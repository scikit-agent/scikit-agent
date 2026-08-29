"""Tests for skagent.ground (GroundedBlock)."""

import numpy as np
import pytest

from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock
from skagent.distributions import MeanOneLogNormal
from skagent.ground import GroundedBlock

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
