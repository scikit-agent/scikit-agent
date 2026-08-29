"""Tests for skagent.ground (GroundedBlock)."""

import numpy as np
import pytest

from skagent.bellman import BellmanPeriod
from skagent.ground import GroundedBlock

from tests.conftest import RECIPE_CALIBRATION, recipe_block


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
