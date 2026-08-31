"""Tests for skagent.solver (solve_multiple_controls)."""

import numpy as np
import pytest
import torch

import skagent.bellman as bellman
import skagent.block as block
import skagent.grid as grid
from skagent.solver import solve_multiple_controls

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CALIBRATION = {"k": 3, "beta": 0.9}


def two_control_period(calibration=None):
    """A static block whose reward is maximized at ``c = a`` and ``d = k``.

    Two controls, one of which conditions on nothing, so the sweep has to solve
    both and neither is solved by the other's pass alone.
    """
    b = block.DBlock(
        name="two controls",
        dynamics={
            "c": block.Control(["a"], agent="agent"),
            "d": block.Control([], agent="agent"),
            "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
        },
        reward={"u": "agent"},
    )
    return bellman.BellmanPeriod(
        b, "beta", dict(CALIBRATION) if calibration is None else calibration
    )


def states():
    return grid.Grid.from_config({"a": {"min": -2, "max": 2, "count": 11}})


class TestSolveMultipleControls:
    def test_every_control_reaches_its_optimum(self):
        torch.manual_seed(TEST_SEED)
        givens = states()

        rules = solve_multiple_controls(
            ["c", "d", "c"], two_control_period(), givens, epochs=100
        )

        a = givens["a"].flatten()
        c = rules["c"](a).detach().cpu().numpy().flatten()
        d = rules["d"]().detach().cpu().numpy().flatten()

        assert np.max(np.abs(c - a.cpu().numpy())) < 0.05
        assert d == pytest.approx(CALIBRATION["k"], abs=0.05)


class TestTheCalibrationArgument:
    """The period already carries a calibration; a second copy may disagree.

    Nothing downstream reconciles the two, so a caller who passes both can hand
    a period one calibration and evaluate its losses at another.
    """

    def test_a_disagreeing_calibration_raises(self):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="'k'"):
            solve_multiple_controls(
                ["c"], two_control_period(), states(), {"k": 4, "beta": 0.9}, epochs=1
            )

    def test_passing_the_calibration_is_deprecated(self):
        torch.manual_seed(TEST_SEED)

        with pytest.warns(DeprecationWarning, match="bellman_period"):
            solve_multiple_controls(
                ["c"], two_control_period(), states(), dict(CALIBRATION), epochs=1
            )
