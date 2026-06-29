from conftest import (
    case_0,
    case_1,
    case_3,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
)
import skagent.algos.vbi as vbi
from skagent.distributions import Bernoulli
from skagent.block import Control, DBlock
from skagent.loss import BellmanEquationLoss
from skagent.grid import device
import skagent.models.consumer as cons
import numpy as np
import xarray as xr
import torch
import unittest


block_1 = DBlock(
    **{
        "name": "vbi_test_1",
        "shocks": {
            "coin": Bernoulli(p=0.5),
        },
        "dynamics": {
            "m": lambda y, coin: y + coin,
            "c": Control(["m"], lower_bound=lambda m: 0, upper_bound=lambda m: m),
            "a": lambda m, c: m - c,
            "u": lambda c: 1 - (c - 1) ** 2,
        },
        "reward": {"u": "agent"},
    }
)

block_2 = DBlock(  # has no control variable
    **{
        "name": "vbi_test_1",
        "shocks": {
            "coin": Bernoulli(p=0.5),
        },
        "dynamics": {
            "m": lambda y, coin: y + coin,
            "a": lambda m: m - 1,
            "u": lambda m: 0,
        },
        "reward": {"u": "agent"},
    }
)


class test_vbi(unittest.TestCase):
    # def setUp(self):
    #    pass

    def test_solve_block_1(self):
        state_grid = {"m": np.linspace(0, 2, 10)}

        dr, dec_vf, arr_vf = vbi.solve(block_1, lambda a: a, state_grid)

        self.assertAlmostEqual(dr["c"](1), 0.5)

    def test_solve_block_2(self):
        # no control variable case.
        state_grid = {"m": np.linspace(0, 2, 10)}

        dr, dec_vf, arr_vf = vbi.solve(block_2, lambda a: a, state_grid)

        # arrival value function gives the correct expect value of continuation
        self.assertAlmostEqual(arr_vf({"y": 10}), 9.5)

    def test_solve_consumption_problem(self):
        state_grid = {"m": np.linspace(0, 5, 10)}

        print(cons.consumption_block_normalized.dynamics["c"])

        dr, dec_vf, arr_vf = vbi.solve(
            cons.consumption_block_normalized,
            lambda a: 0,
            state_grid,
            disc_params={"theta": {"N": 7}},
            scope=cons.calibration,
        )

        self.assertAlmostEqual(dr["c"](1.5), 1.5)


# Terminal continuation: the value of arriving at the next block is zero.
# With this, each conftest case reduces to a single backward-induction step
# whose optimum is the case's documented ``optimal_dr``.
def terminal_continuation(a):
    return 0.0


class test_vbi_conftest(unittest.TestCase):
    """
    Comprehensive backward-induction tests against the shared conftest suite.

    Each case ships an analytic ``optimal_dr``; here we solve the block with
    VBI and check the recovered decision rule gets close to that optimum at
    interior points of the state grid. Together these exercise the full range
    of the single-control solver: an interior optimum, a shock-dependent
    policy, both-sided bounds (with either side binding), single-sided bounds
    (which lean on the open-bound defaults), and an empty information set.
    """

    # tolerance on the recovered policy; the optima here are all linear, so
    # grid interpolation is exact and scipy's optimizer is the only error source
    ATOL = 1e-3

    def test_case_0_interior_optimum(self):
        # u = -c^2, unconstrained -> c* = 0 for all a
        state_grid = {"a": np.linspace(0, 2, 11)}
        dr, _, _ = vbi.solve(
            case_0["block"],
            terminal_continuation,
            state_grid,
            scope=case_0["calibration"],
        )
        for a in [0.2, 0.7, 1.3, 1.8]:
            self.assertAlmostEqual(dr["c"](a), 0.0, delta=self.ATOL)

    def test_case_1_shock_dependent_policy(self):
        # u = -(theta - c)^2 with theta in the information set -> c* = theta
        state_grid = {
            "a": np.linspace(0, 1, 7),
            "theta": np.linspace(-1, 1, 7),
        }
        dr, _, _ = vbi.solve(
            case_1["block"],
            terminal_continuation,
            state_grid,
            scope=case_1["calibration"],
        )
        for theta in [-0.6, 0.0, 0.4, 0.9]:
            self.assertAlmostEqual(dr["c"](0.5, theta), theta, delta=self.ATOL)

    def test_case_3_consume_cash_on_hand(self):
        # u = -(m - c)^2 -> c* = m. The grid is just the iset, [m]. The arrival
        # state ``a`` depends on the psi shock, so psi is supplied via the
        # calibration (it only enters the transition, not the decision).
        state_grid = {"m": np.linspace(0.1, 2, 7)}
        dr, _, _ = vbi.solve(
            case_3["block"],
            terminal_continuation,
            state_grid,
            scope={**case_3["calibration"], "psi": 0.0},
        )
        for m in [0.5, 1.0, 1.5]:
            self.assertAlmostEqual(dr["c"](m), m, delta=self.ATOL)

    def test_case_5_double_bounded_upper_binds(self):
        # maximize c subject to 0 <= c <= a -> c* = a (upper bound binds)
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, _ = vbi.solve(
            case_5["block"],
            terminal_continuation,
            state_grid,
            scope=case_5["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_6_double_bounded_lower_binds(self):
        # minimize c subject to a <= c <= 2a -> c* = a (lower bound binds)
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, _ = vbi.solve(
            case_6["block"],
            terminal_continuation,
            state_grid,
            scope=case_6["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_7_only_lower_bound(self):
        # minimize c subject to c >= 1 (no upper bound) -> c* = 1.
        # Exercises the open upper-bound default.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, _ = vbi.solve(
            case_7["block"],
            terminal_continuation,
            state_grid,
            scope=case_7["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), 1.0, delta=self.ATOL)

    def test_case_8_only_upper_bound(self):
        # maximize c subject to c <= a (no lower bound) -> c* = a.
        # Exercises the open lower-bound default.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, _ = vbi.solve(
            case_8["block"],
            terminal_continuation,
            state_grid,
            scope=case_8["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_9_empty_information_set(self):
        # u = -(c - 3)^2 with an empty information set -> constant c* = 3.
        # The iset is empty, so the grid is empty too (contract: grid == iset).
        # The arrival state ``a`` (which the continuation ranges over) is value-
        # irrelevant under terminal continuation, so it is supplied via the
        # calibration rather than as a grid axis.
        state_grid = {}
        dr, _, _ = vbi.solve(
            case_9["block"],
            terminal_continuation,
            state_grid,
            scope={**case_9["calibration"], "a": 0.0},
        )
        # empty iset -> the rule is constant across the grid
        self.assertTrue(np.allclose(dr["c"](), 3.0, atol=self.ATOL))


# Terminal continuation on the BellmanPeriod convention: V'(s') = 0 for all
# next-period arrival states, shocks, and parameters. Distinct from the legacy
# one-argument ``terminal_continuation`` above (which rides the DBlock API).
def bp_terminal(states, shocks, parameters):
    return 0.0


class test_vbi_bellman_step(unittest.TestCase):
    """
    PR1 of the Phase-2 design (§9 step 1): ``vbi.bellman_step`` — one exact value
    backup on the ``BellmanPeriod`` protocol, single control, grid-equals-iset.

    Under a terminal (zero) continuation each conftest case reduces to a single
    backward-induction step whose optimum is the case's analytic ``optimal_dr``.
    These mirror ``test_vbi_conftest`` (which exercises legacy ``solve``) but
    drive ``bellman_step`` and assert its 3-tuple return contract.
    """

    # The optima here are all linear, so grid interpolation is exact and scipy's
    # optimizer is the only error source.
    ATOL = 1e-3

    def _step(self, case, state_grid, scope):
        return vbi.bellman_step(case["bp"], bp_terminal, state_grid, scope=scope)

    def test_case_0_interior_optimum(self):
        # u = -c^2, unconstrained -> c* = 0 for all a
        dr, _, _ = self._step(
            case_0, {"a": np.linspace(0, 2, 11)}, case_0["calibration"]
        )
        for a in [0.2, 0.7, 1.3, 1.8]:
            self.assertAlmostEqual(dr["c"](a), 0.0, delta=self.ATOL)

    def test_case_1_shock_dependent_policy(self):
        # u = -(theta - c)^2 with theta an OBSERVED shock in the iset -> c* = theta
        dr, _, _ = self._step(
            case_1,
            {"a": np.linspace(0, 1, 7), "theta": np.linspace(-1, 1, 7)},
            case_1["calibration"],
        )
        for theta in [-0.6, 0.0, 0.4, 0.9]:
            self.assertAlmostEqual(dr["c"](0.5, theta), theta, delta=self.ATOL)

    def test_case_5_double_bounded_upper_binds(self):
        # maximize c subject to 0 <= c <= a -> c* = a. theta is a HIDDEN shock
        # (only in the transition); supply a fixed realization via scope (PR1).
        dr, _, _ = self._step(
            case_5,
            {"a": np.linspace(0.2, 1, 5)},
            {**case_5["calibration"], "theta": 0.0},
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_6_double_bounded_lower_binds(self):
        # minimize c subject to a <= c <= 2a -> c* = a (lower bound binds)
        dr, _, _ = self._step(
            case_6,
            {"a": np.linspace(0.2, 1, 5)},
            {**case_6["calibration"], "theta": 0.0},
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_7_only_lower_bound(self):
        # minimize c subject to c >= 1 (no upper bound) -> c* = 1.
        # Exercises the open upper-bound default and the x0 fallback seed.
        dr, _, _ = self._step(
            case_7,
            {"a": np.linspace(0.2, 1, 5)},
            {**case_7["calibration"], "theta": 0.0},
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), 1.0, delta=self.ATOL)

    def test_case_8_only_upper_bound(self):
        # maximize c subject to c <= a (no lower bound) -> c* = a.
        # Exercises the open lower-bound default.
        dr, _, _ = self._step(
            case_8,
            {"a": np.linspace(0.2, 1, 5)},
            {**case_8["calibration"], "theta": 0.0},
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)

    def test_case_9_empty_information_set(self):
        # u = -(c - 3)^2 with an empty iset -> constant c* = 3. Grid is empty
        # (grid == iset); the value-irrelevant arrival state a goes in scope.
        dr, _, policy = self._step(case_9, {}, {**case_9["calibration"], "a": 0.0})
        self.assertTrue(np.allclose(dr["c"](), 3.0, atol=self.ATOL))
        # empty iset -> 0-dimensional policy array
        self.assertEqual(policy["c"].ndim, 0)

    def test_return_contract(self):
        # value_array is a DataArray over the grid; policy_array is a dict of
        # DataArrays keyed by control symbol (O1).
        grid = {"a": np.linspace(0, 2, 11)}
        dr, value_array, policy_array = self._step(case_0, grid, case_0["calibration"])
        self.assertIsInstance(value_array, xr.DataArray)
        self.assertEqual(list(value_array.dims), ["a"])
        self.assertIsInstance(policy_array, dict)
        self.assertIsInstance(policy_array["c"], xr.DataArray)
        self.assertEqual(list(policy_array["c"].dims), ["a"])
        # terminal continuation: V(a) = max_c -c^2 = 0
        self.assertTrue(np.allclose(value_array.values, 0.0, atol=self.ATOL))
        # the gridded policy matches the fitted rule at the nodes
        self.assertTrue(np.allclose(policy_array["c"].values, 0.0, atol=self.ATOL))

    def test_warm_start_x0_policy(self):
        # Passing a previous iterate's policy_array as x0_policy seeds the
        # optimizer per point and reproduces the same optimum (the path
        # solve_bellman uses across iterations).
        grid = {"a": np.linspace(0, 2, 11)}
        _, _, policy1 = self._step(case_0, grid, case_0["calibration"])
        _, _, policy2 = vbi.bellman_step(
            case_0["bp"],
            bp_terminal,
            grid,
            scope=case_0["calibration"],
            x0_policy=policy1,
        )
        self.assertTrue(
            np.allclose(policy1["c"].values, policy2["c"].values, atol=self.ATOL)
        )

    def test_multi_control_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vbi.bellman_step(case_10["bp"], bp_terminal, {"a": np.linspace(-2, 2, 11)})

    def test_disc_params_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vbi.bellman_step(
                case_0["bp"],
                bp_terminal,
                {"a": np.linspace(0, 2, 5)},
                disc_params={"theta": {"N": 3}},
            )


class test_vbi_protocol(unittest.TestCase):
    """
    Phase 1 deliverable: a VBI-fitted decision rule is a drop-in for the
    torch-based ``BellmanPeriod`` stack.

    ``vbi.solve`` returns numpy/xarray-space decision rules (positional, in
    information-set order). ``vbi.tensor_decision_rule`` wraps each so it
    accepts and returns torch tensors. The wrapped dict then flows, unmodified,
    through ``BellmanPeriod.compute_controls`` and ``BellmanEquationLoss``.
    """

    def _solve_tensor_dr(self, case, state_grid):
        dr, _, _ = vbi.solve(
            case["block"],
            terminal_continuation,
            state_grid,
            scope=case["calibration"],
        )
        return {c: vbi.tensor_decision_rule(rule) for c, rule in dr.items()}

    def test_compute_controls_roundtrip(self):
        # case_0: u = -c^2, iset = [a] -> c* = 0. The tensorized VBI rule, fed
        # to BellmanPeriod.compute_controls as a dict of decision rules, returns
        # a float32 tensor of optimal controls on the batch of states.
        dr_t = self._solve_tensor_dr(case_0, {"a": np.linspace(0, 2, 11)})

        a = torch.linspace(0.2, 1.8, 5, device=device)
        controls = case_0["bp"].compute_controls(
            dr_t, {"a": a}, shocks={}, parameters=case_0["calibration"]
        )

        self.assertIn("c", controls)
        self.assertEqual(controls["c"].dtype, torch.float32)
        self.assertEqual(controls["c"].shape, a.shape)
        self.assertTrue(torch.allclose(controls["c"], torch.zeros_like(a), atol=1e-3))

    def test_bellman_equation_loss_roundtrip(self):
        # case_1: u = -(theta - c)^2, iset = [a, theta] -> c* = theta. Under a
        # zero continuation the VBI policy is exactly optimal, so V(s) = 0 and
        # the Bellman residual u + beta*E[V(s')] - V(s) vanishes. We only assert
        # the loss is finite and (here) ~0 -- the point is that the VBI dr is a
        # valid ``df`` for BellmanEquationLoss.
        dr_t = self._solve_tensor_dr(
            case_1,
            {"a": np.linspace(0, 1, 7), "theta": np.linspace(-1, 1, 7)},
        )

        def zero_value_function(states, shocks, parameters):
            return torch.zeros_like(states["a"])

        loss_fn = BellmanEquationLoss(
            case_1["bp"], zero_value_function, parameters=case_1["calibration"]
        )
        # givens[2] carries a, theta_0, theta_1 (two independent shock draws)
        loss = loss_fn(dr_t, case_1["givens"][2])

        self.assertTrue(torch.isfinite(loss).all())
        self.assertLess(float(loss.mean()), 1e-3)
