from conftest import (
    case_0,
    case_1,
    case_2,
    case_3,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
    case_11,
    count_calls,
)
import skagent.algos.vfi as vfi
from skagent.bellman import BellmanPeriod
from skagent.distributions import Bernoulli
from skagent.block import Control, DBlock
from skagent.loss import BellmanEquationLoss
from skagent.grid import device
import skagent.models.benchmarks as bm
import skagent.models.consumer as cons
import skagent.models.fisher as fisher
import numpy as np
import xarray as xr
import torch
import unittest
import warnings


block_1 = DBlock(
    **{
        "name": "vfi_test_1",
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
        "name": "vfi_test_1",
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


class test_vfi(unittest.TestCase):
    # def setUp(self):
    #    pass

    def test_solve_block_1(self):
        state_grid = {"m": np.linspace(0, 2, 10)}

        dr, dec_vf, arr_vf = vfi.solve(block_1, lambda a: a, state_grid)

        self.assertAlmostEqual(dr["c"](1), 0.5)

    def test_solve_block_2(self):
        # no control variable case.
        state_grid = {"m": np.linspace(0, 2, 10)}

        dr, dec_vf, arr_vf = vfi.solve(block_2, lambda a: a, state_grid)

        # arrival value function gives the correct expect value of continuation
        self.assertAlmostEqual(arr_vf({"y": 10}), 9.5)

    def test_solve_consumption_problem(self):
        state_grid = {"m": np.linspace(0, 5, 10)}

        print(cons.consumption_block_normalized.dynamics["c"])

        dr, dec_vf, arr_vf = vfi.solve(
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


class test_vfi_conftest(unittest.TestCase):
    """
    Comprehensive backward-induction tests against the shared conftest suite.

    Each case ships an analytic ``optimal_dr``; here we solve the block with
    VFI and check the recovered decision rule gets close to that optimum at
    interior points of the state grid. Together these exercise the full range
    of the single-control solver: an interior optimum, a shock-dependent
    policy, both-sided bounds (with either side binding), single-sided bounds
    (which lean on the open-bound defaults), and an empty information set.

    Each case solves *once* and checks both value functions ``solve`` returns
    alongside the policy, so the value functions are covered without re-running
    the solver. Under ``terminal_continuation`` (a zero continuation) the value
    at the optimum is just the period reward ``u(c*)``, giving a closed form:

    - ``dec_vf(pre)`` is the decision-node value (after shocks); it runs the
      full transition, so it is checked on the shock-free / shock-in-iset cases
      (0, 1, 9) where every variable the transition needs is available.
    - ``arr_vf(arrival)`` is the arrival value; it discretizes the block's
      shocks and integrates ``dec_vf`` over them. It is checked on the bounded
      cases (5-8) whose reward does not depend on the shock, so the expectation
      collapses to the analytic value. ``disc_params`` only feeds ``arr_vf``
      construction and does not affect the policy solve (VFI is full-observation
      and never integrates over shocks in its per-point optimization).
    """

    # tolerance on the recovered policy; the optima here are all linear, so
    # grid interpolation is exact and scipy's optimizer is the only error source
    ATOL = 1e-3

    def test_case_0_interior_optimum(self):
        # u = -c^2, unconstrained -> c* = 0 for all a; V(a) = 0.
        state_grid = {"a": np.linspace(0, 2, 11)}
        dr, dec_vf, _ = vfi.solve(
            case_0["block"],
            terminal_continuation,
            state_grid,
            scope=case_0["calibration"],
        )
        for a in [0.2, 0.7, 1.3, 1.8]:
            self.assertAlmostEqual(dr["c"](a), 0.0, delta=self.ATOL)
            self.assertAlmostEqual(
                float(dec_vf({**case_0["calibration"], "a": a})), 0.0, delta=self.ATOL
            )

    def test_case_1_shock_dependent_policy(self):
        # u = -(theta - c)^2 with theta in the information set -> c* = theta;
        # V(a, theta) = 0. theta is in the iset, so dec_vf conditions on it.
        state_grid = {
            "a": np.linspace(0, 1, 7),
            "theta": np.linspace(-1, 1, 7),
        }
        dr, dec_vf, _ = vfi.solve(
            case_1["block"],
            terminal_continuation,
            state_grid,
            scope=case_1["calibration"],
        )
        for theta in [-0.6, 0.0, 0.4, 0.9]:
            self.assertAlmostEqual(dr["c"](0.5, theta), theta, delta=self.ATOL)
            self.assertAlmostEqual(
                float(dec_vf({**case_1["calibration"], "a": 0.5, "theta": theta})),
                0.0,
                delta=self.ATOL,
            )

    def test_case_3_consume_cash_on_hand(self):
        # u = -(m - c)^2 -> c* = m. The grid is just the iset, [m]. The arrival
        # state ``a`` depends on the psi shock, so psi is supplied via the
        # calibration (it only enters the transition, not the decision).
        #
        # Policy-only: the value functions are awkward here because ``m`` is a
        # computed intermediate (m = a + theta), so evaluating dec_vf/arr_vf
        # would require the pre-``m`` arrival state, not the [m] iset the rule
        # conditions on. The other cases cover the value functions.
        state_grid = {"m": np.linspace(0.1, 2, 7)}
        dr, _, _ = vfi.solve(
            case_3["block"],
            terminal_continuation,
            state_grid,
            scope={**case_3["calibration"], "psi": 0.0},
        )
        for m in [0.5, 1.0, 1.5]:
            self.assertAlmostEqual(dr["c"](m), m, delta=self.ATOL)

    def test_case_5_double_bounded_upper_binds(self):
        # maximize c subject to 0 <= c <= a -> c* = a (upper bound binds);
        # V(a) = a. theta only enters next period's a, so arr_vf collapses to a.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, arr_vf = vfi.solve(
            case_5["block"],
            terminal_continuation,
            state_grid,
            disc_params={"theta": {"N": 7}},
            scope=case_5["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)
            self.assertAlmostEqual(float(arr_vf({"a": a})), a, delta=self.ATOL)

    def test_case_6_double_bounded_lower_binds(self):
        # minimize c subject to a <= c <= 2a -> c* = a (lower bound binds);
        # u = -c, so V(a) = -a.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, arr_vf = vfi.solve(
            case_6["block"],
            terminal_continuation,
            state_grid,
            disc_params={"theta": {"N": 7}},
            scope=case_6["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)
            self.assertAlmostEqual(float(arr_vf({"a": a})), -a, delta=self.ATOL)

    def test_case_7_only_lower_bound(self):
        # minimize c subject to c >= 1 (no upper bound) -> c* = 1.
        # Exercises the open upper-bound default. u = -c, so V(a) = -1.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, arr_vf = vfi.solve(
            case_7["block"],
            terminal_continuation,
            state_grid,
            disc_params={"theta": {"N": 7}},
            scope=case_7["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), 1.0, delta=self.ATOL)
            self.assertAlmostEqual(float(arr_vf({"a": a})), -1.0, delta=self.ATOL)

    def test_case_8_only_upper_bound(self):
        # maximize c subject to c <= a (no lower bound) -> c* = a.
        # Exercises the open lower-bound default. u = c, so V(a) = a.
        state_grid = {"a": np.linspace(0.2, 1, 5)}
        dr, _, arr_vf = vfi.solve(
            case_8["block"],
            terminal_continuation,
            state_grid,
            disc_params={"theta": {"N": 7}},
            scope=case_8["calibration"],
        )
        for a in [0.4, 0.6, 0.9]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)
            self.assertAlmostEqual(float(arr_vf({"a": a})), a, delta=self.ATOL)

    def test_case_9_empty_information_set(self):
        # u = -(c - 3)^2 with an empty information set -> constant c* = 3.
        # The iset is empty, so the grid is empty too (contract: grid == iset).
        # The arrival state ``a`` (which the continuation ranges over) is value-
        # irrelevant under terminal continuation, so it is supplied via the
        # calibration rather than as a grid axis.
        state_grid = {}
        dr, dec_vf, _ = vfi.solve(
            case_9["block"],
            terminal_continuation,
            state_grid,
            scope={**case_9["calibration"], "a": 0.0},
        )
        # empty iset -> the rule is constant across the grid
        self.assertTrue(np.allclose(dr["c"](), 3.0, atol=self.ATOL))
        # V = -(c* - 3)^2 = 0 at the (constant) optimum
        self.assertAlmostEqual(
            float(dec_vf({**case_9["calibration"], "a": 0.0})), 0.0, delta=self.ATOL
        )


# Terminal continuation on the BellmanPeriod convention: V'(s') = 0 for all
# next-period arrival states, shocks, and parameters. Distinct from the legacy
# one-argument ``terminal_continuation`` above (which rides the DBlock API).
def bp_terminal(states, shocks, parameters):
    return 0.0


class test_vfi_bellman_step(unittest.TestCase):
    """
    Phase-2 design: ``vfi.bellman_step`` — one exact value backup
    on the ``BellmanPeriod`` protocol; single- and multi-control (one joint
    ``scipy.minimize`` over the stacked control vector, per-control iset
    projection).

    Under a terminal (zero) continuation each conftest case reduces to a single
    backward-induction step whose optimum is the case's analytic ``optimal_dr``.
    These mirror ``test_vfi_conftest`` (which exercises legacy ``solve``) but
    drive ``bellman_step`` and assert its 3-tuple return contract. The
    Mechanism-B reindex is also exercised: a control whose information set
    is a derived pre-state (``case_3``'s ``m = a + theta``, D-2's ``m = a·R + y``).
    """

    # The optima here are all linear, so grid interpolation is exact and scipy's
    # optimizer is the only error source.
    ATOL = 1e-3

    def _step(self, case, state_grid, scope):
        return vfi.bellman_step(case["bp"], bp_terminal, state_grid, scope=scope)

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
        # (only in the transition); supply a fixed realization via scope.
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

    def test_one_block_pass_per_objective_evaluation(self):
        # The per-point objective reads the reward, the discount factor and the
        # next arrival states off one ex post result, so a backup runs the
        # dynamics once per objective evaluation rather than three times.
        # case_0's control conditions on the arrival state itself, so no
        # pre-state pass is needed and every pass is an objective evaluation.
        with count_calls(case_0["bp"].block, "transition") as passes:
            with count_calls(case_0["bp"], "post_function") as objectives:
                self._step(case_0, {"a": np.linspace(0, 2, 3)}, case_0["calibration"])
        self.assertGreater(objectives["n"], 0)
        self.assertEqual(passes["n"], objectives["n"])

    def test_warm_start_x0_policy(self):
        # Passing a previous iterate's policy_array as x0_policy seeds the
        # optimizer per point and reproduces the same optimum (the path
        # solve_bellman uses across iterations).
        grid = {"a": np.linspace(0, 2, 11)}
        _, _, policy1 = self._step(case_0, grid, case_0["calibration"])
        _, _, policy2 = vfi.bellman_step(
            case_0["bp"],
            bp_terminal,
            grid,
            scope=case_0["calibration"],
            x0_policy=policy1,
        )
        self.assertTrue(
            np.allclose(policy1["c"].values, policy2["c"].values, atol=self.ATOL)
        )

    def test_case_10_multi_control(self):
        # Two controls with DIFFERENT information sets, jointly optimized by a
        # single scipy.minimize over the stacked [c, d] vector:
        #   c.iset = [a] -> c* = a   (grid equals iset, transpose projection)
        #   d.iset = []  -> d* = k=3 (Mechanism-A reduction drops the a axis)
        # u = -(a-c)^2 - (k-d)^2 is separable, so the optima are independent.
        dr, _, policy = self._step(
            case_10, {"a": np.linspace(-2, 2, 11)}, case_10["calibration"]
        )
        for a in [-1.5, -0.5, 0.5, 1.5]:
            self.assertAlmostEqual(dr["c"](a), a, delta=self.ATOL)
        # d's iset is empty -> a constant rule recovered via Mechanism A.
        self.assertTrue(np.allclose(dr["d"](), 3.0, atol=self.ATOL))
        # policy_array carries BOTH controls (O1), each over the state grid so
        # solve_bellman can warm-start; the per-control iset projection is only
        # applied to the decision rules.
        self.assertEqual(set(policy), {"c", "d"})
        self.assertEqual(list(policy["c"].dims), ["a"])
        self.assertEqual(list(policy["d"].dims), ["a"])

    def test_case_11_nontrivial_continuation(self):
        # A real continuation_vf drives the optimum. The period reward
        # u = -(a - b)^2 is over the ARRIVAL states (a, b) and is
        # independent of the control c, so the immediate reward alone cannot pin
        # c. The transition carries c forward as next-period b' (b' = c) while
        # a' = a + theta; a continuation that rewards b' ~ a' therefore pulls
        #   c* = a' = a + theta.
        # This exercises beta*cv and the arrival transition together: the optimum
        # exists only because of the discounted continuation value.
        def continuation(states, shocks, parameters):
            return -((states["a"] - states["b"]) ** 2)

        grid = {
            "a": np.linspace(-1.5, 1.5, 7),
            "b": np.linspace(-1.5, 1.5, 5),  # outside c.iset = [a, theta]
            "theta": np.linspace(-1, 1, 5),
        }
        dr, value_array, policy = vfi.bellman_step(
            case_11["bp"], continuation, grid, agent="agent"
        )
        # c.iset = [a, theta]; the b axis is outside it and the optimum is
        # invariant along it, so Mechanism A drops it -> rule of (a, theta).
        for a in [-1.0, 0.0, 1.0]:
            for theta in [-0.5, 0.5]:
                self.assertAlmostEqual(dr["c"](a, theta), a + theta, delta=self.ATOL)
        # At the optimum the continuation is driven to zero (b' = c = a + theta =
        # a'), so the decision value is just the arrival reward V = -(a - b)^2 --
        # confirming the reward reads the arrival b, not the control.
        self.assertEqual(list(value_array.dims), ["a", "b", "theta"])
        a_g, b_g = np.meshgrid(grid["a"], grid["b"], indexing="ij")
        want_v = -((a_g[:, :, None] - b_g[:, :, None]) ** 2) * np.ones(
            (1, 1, len(grid["theta"]))
        )
        self.assertTrue(np.allclose(value_array.values, want_v, atol=self.ATOL))

    def test_project_to_iset_non_invariant_raises(self):
        # Dropping a grid axis outside the iset assumes the optimum is invariant
        # along it; a policy that actually varies there must fail loudly.
        policy = xr.DataArray(
            np.array([0.0, 1.0, 2.0]), dims=["a"], coords={"a": [0.0, 1.0, 2.0]}
        )
        with self.assertRaises(ValueError):
            vfi._project_to_iset(policy, ["a"], [], {}, "c")

    def test_case_2_hidden_shock_expectation(self):
        # u = -(theta - c)^2 with theta a HIDDEN shock (iset = [a], so theta is
        # in no control's information set). The backup integrates theta out
        # inside the max: E_theta[-(theta - c)^2] = -(Var[theta] + (E[theta]-c)^2)
        # is maximized at c = E[theta] = 0, independent of a. This is the minimal
        # unit test of the hidden-shock discretization.
        dr, value_array, _ = vfi.bellman_step(
            case_2["bp"],
            bp_terminal,
            {"a": np.linspace(0, 1, 5)},
            scope=case_2["calibration"],
            disc_params={"theta": {"N": 7}},
        )
        for a in [0.2, 0.5, 0.8]:
            self.assertAlmostEqual(dr["c"](a), 0.0, delta=self.ATOL)
        # At the optimum c = E[theta] the value is -Var[theta] = -1 (standard
        # normal), the irreducible variance the agent cannot hedge.
        self.assertTrue(np.allclose(value_array.values, -1.0, atol=1e-2))

    def test_u2_single_backup_analytic_continuation(self):
        # U-2 (log utility, normalized permanent-income shock psi, no borrowing
        # constraint): psi is a HIDDEN shock (c.iset = [m], m = R*a/psi + 1). At
        # the default sigma_psi = 0, psi discretizes to a single degenerate node
        # at psi = 1, so the hidden-shock expectation is exact. Under the exact
        # log-utility arrival value function, a single backup recovers the PIH
        # policy c = (1 - beta)(m + 1/r). Exercises hidden-shock integration
        # with a pre-state (m) that depends on the (degenerate) hidden shock.
        cal = bm.u2_calibration
        beta, R = cal["DiscFac"], cal["R"]
        h = 1.0 / (R - 1.0)  # human wealth (normalized)
        # V(W) = A + B*log(W), B = 1/(1-beta), on total wealth W = m + h. The
        # additive constant A does not shift argmax_c, so it is dropped. At an
        # arrival state a', next-period total wealth is W' = R*(a' + h).
        B = 1.0 / (1.0 - beta)

        def u2_continuation(states, shocks, parameters):
            return B * np.log(R * (states["a"] + h))

        bp = BellmanPeriod(bm.u2_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 12)}
        dr, _, _ = vfi.bellman_step(bp, u2_continuation, grid, scope=cal)
        for a in [1.0, 2.0, 3.0]:
            m = R * a + 1.0  # psi = 1 at sigma_psi = 0
            want = float(
                bm.u2_analytical_policy({"a": torch.tensor(float(a))}, {}, cal)["c"]
            )
            self.assertAlmostEqual(dr["c"](m), want, delta=self.ATOL)

    def test_u2_multinode_recovers_closed_form(self):
        # sigma_psi > 0 spreads psi over several discretization nodes. Because psi
        # reaches the objective only through m, which c conditions on, each node
        # gets its own pre-state and bounds and the backup solves the problem the
        # block declares -- so the PIH closed form c = (1-beta)(m + 1/r) is
        # recovered at sigma > 0, not only at the degenerate sigma = 0.
        beta, R = bm.u2_calibration["DiscFac"], bm.u2_calibration["R"]
        h = 1.0 / (R - 1.0)
        B = 1.0 / (1.0 - beta)

        def u2_continuation(states, shocks, parameters):
            # additive constant dropped; irrelevant to argmax_c (see above test)
            return B * np.log(R * (states["a"] + h))

        grid = {"a": np.linspace(0.5, 8.0, 16)}
        ms = [R * a + 1.0 for a in [1.0, 2.0, 3.0, 5.0]]

        def solve(sigma_psi):
            cal = {**bm.u2_calibration, "sigma_psi": sigma_psi}
            bp = BellmanPeriod(bm.u2_block, "DiscFac", cal)
            dr, _, _ = vfi.bellman_step(
                bp, u2_continuation, grid, scope=cal, disc_params={"psi": {"N": 7}}
            )
            return np.array([dr["c"](m) for m in ms])

        want = np.array([(1.0 - beta) * (m + h) for m in ms])
        for sigma_psi in (0.0, 0.1, 0.2):
            got = solve(sigma_psi)
            self.assertTrue(np.all(got > 0))
            self.assertTrue(np.all(np.diff(got) > 0))
            np.testing.assert_allclose(got, want, atol=1e-4)

    def test_u2_rule_over_m_does_not_depend_on_shock_spread(self):
        # The decision problem at a given m -- max_c u(c) + beta*W(m - c) -- does
        # not involve sigma_psi, so the rule as a function of m is invariant to it;
        # only the distribution of m changes. Fixing psi at its mean instead puts
        # the expectation inside the max and makes the rule spuriously
        # sigma-dependent, so this pins the ordering of max and expectation.
        beta, R = bm.u2_calibration["DiscFac"], bm.u2_calibration["R"]
        h = 1.0 / (R - 1.0)
        B = 1.0 / (1.0 - beta)

        def u2_continuation(states, shocks, parameters):
            return B * np.log(R * (states["a"] + h))

        grid = {"a": np.linspace(0.5, 8.0, 16)}
        ms = [R * a + 1.0 for a in [1.0, 2.0, 3.0, 5.0]]

        def solve(sigma_psi):
            cal = {**bm.u2_calibration, "sigma_psi": sigma_psi}
            bp = BellmanPeriod(bm.u2_block, "DiscFac", cal)
            dr, _, _ = vfi.bellman_step(
                bp, u2_continuation, grid, scope=cal, disc_params={"psi": {"N": 7}}
            )
            return np.array([dr["c"](m) for m in ms])

        np.testing.assert_allclose(solve(0.2), solve(0.0), atol=1e-4)

    def test_u1_continuous_shock_recovers_pih_closed_form(self):
        # U-1 (Hall random walk): eta ~ Normal reaches the objective only through
        # the pre-state m = R*A + y_mean + eta that c conditions on, so it becomes
        # a Gauss-Hermite node axis and m varies along both A and eta. Under the
        # analytic PIH continuation a single backup recovers c = (r/R)(m + H)
        # exactly. The only benchmark-level exercise of a continuous shock, and so
        # of disc_params.
        cal = bm.get_benchmark_calibration("U-1")
        quad_a, quad_b = cal["quad_a"], cal["quad_b"]
        R, y_mean = cal["R"], cal["y_mean"]
        r = R - 1.0
        H = y_mean / r  # present value of the expected income stream
        kappa = r / R  # annuity factor

        def u1_continuation(states, shocks, parameters):
            # V(m) = a*m - (b*kappa/2)(m + H)^2 integrated over next period's
            # income; the Var(eta) term is constant in the control and dropped.
            m_next = R * states["A"] + y_mean
            return quad_a * m_next - (quad_b * kappa / 2.0) * (m_next + H) ** 2

        bp = BellmanPeriod(bm.u1_block, "DiscFac", cal)
        dr, _, _ = vfi.bellman_step(
            bp,
            u1_continuation,
            {"A": np.linspace(0.5, 6.0, 14)},
            scope=cal,
            disc_params={"eta": {"N": 7}},
        )
        for A in [1.0, 2.0, 3.0, 5.0]:
            m = R * A + y_mean
            self.assertAlmostEqual(dr["c"](m), kappa * (m + H), delta=1e-4)

    def test_u3_two_prestate_shocks_degenerate_limit(self):
        # U-3 has *two* shocks feeding the pre-state m = R*a/psi + theta, so both
        # become node axes and m varies along all three grid axes -- and m pins
        # down neither shock individually, which is where the gather-and-fit
        # consistency check is load-bearing rather than vacuous.
        #
        # U-3 has no closed form in general. At sigma_theta = 0 (theta collapses to
        # a point mass at 1) with CRRA = 1 it *is* U-2, so the PIH closed form
        # applies while the two-shock joint node axis is still exercised.
        cal = {
            **bm.get_benchmark_calibration("U-3"),
            "CRRA": 1.0,
            "sigma_theta": 0.0,
        }
        beta, R = cal["DiscFac"], cal["R"]
        h = 1.0 / (R - 1.0)
        B = 1.0 / (1.0 - beta)

        def u3_continuation(states, shocks, parameters):
            return B * np.log(R * (states["a"] + h))

        bp = BellmanPeriod(bm.u3_block, "DiscFac", cal)
        dr, _, _ = vfi.bellman_step(
            bp,
            u3_continuation,
            {"a": np.linspace(0.5, 8.0, 14)},
            scope=cal,
            disc_params={"psi": {"N": 5}, "theta": {"N": 5}},
        )
        for A in [1.0, 2.0, 3.0, 5.0]:
            m = R * A + 1.0
            self.assertAlmostEqual(dr["c"](m), (1.0 - beta) * (m + h), delta=1e-4)

    def test_u3_two_prestate_shocks_properties(self):
        # U-3 at its own calibration: CRRA = 2 and a genuinely spread theta, so
        # both shocks are non-degenerate node axes. No closed form exists, so this
        # asserts only what does not depend on the supplied continuation being the
        # model's own: the rule is positive, non-decreasing in cash-on-hand, and
        # respects the block's borrowing constraint c <= m.
        cal = bm.get_benchmark_calibration("U-3")
        R, sigma = cal["R"], cal["CRRA"]
        h = 1.0 / (R - 1.0)
        B = 1.0 / (1.0 - cal["DiscFac"])

        def u3_continuation(states, shocks, parameters):
            wealth = np.maximum(R * (states["a"] + h), 1e-8)
            return B * wealth ** (1 - sigma) / (1 - sigma)

        bp = BellmanPeriod(bm.u3_block, "DiscFac", cal)
        dr, _, _ = vfi.bellman_step(
            bp,
            u3_continuation,
            {"a": np.linspace(0.5, 8.0, 14)},
            scope=cal,
            disc_params={"psi": {"N": 5}, "theta": {"N": 5}},
        )
        ms = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
        c = np.array([dr["c"](m) for m in ms])
        self.assertTrue(np.all(c > 0))
        self.assertTrue(np.all(np.diff(c) > 0))
        self.assertTrue(np.all(c <= ms + self.ATOL))

    # --- iset is a derived pre-state: reproject onto its coordinate ----

    def test_case_3_derived_iset_reproject(self):
        # u = -(m - c)^2 with iset = [m], m = a + theta a derived pre-state. The
        # grid is over the arrival state a (theta, psi fixed in scope, so the
        # map a -> m = a + theta is 1-D and strictly monotone); bellman_step
        # reindexes the policy onto the m coordinate -> c* = m. theta is held at
        # a non-zero value so the m axis genuinely differs from the a axis.
        theta0 = 0.5
        dr, _, policy = self._step(
            case_3,
            {"a": np.linspace(0.1, 2.0, 8)},
            {**case_3["calibration"], "theta": theta0, "psi": 0.0},
        )
        # m ranges over [0.1 + theta0, 2.0 + theta0]; probe interior values.
        for m in [1.0, 1.5, 2.0]:
            self.assertAlmostEqual(dr["c"](m), m, delta=self.ATOL)
        # policy_array stays over the *state grid* (axis a), for warm-starting;
        # only the decision rule moves to the m coordinate.
        self.assertEqual(list(policy["c"].dims), ["a"])

    def test_d2_single_backup_analytic_continuation(self):
        # D-2 (infinite-horizon CRRA, no shocks): a single backup under the
        # *exact* arrival value function recovers the analytic policy
        # c = kappa*(m + H). Exercises Mechanism B with m = a*R + y and a
        # non-trivial continuation, decoupled from the iteration loop.
        cal = bm.d2_calibration
        beta, R, sigma, y = cal["DiscFac"], cal["R"], cal["CRRA"], cal["y"]
        H = y / (R - 1)  # human wealth
        kappa = (R - (beta * R) ** (1 / sigma)) / R
        # Closed-form CRRA value in total wealth W: with c = kappa*W and
        # W' = (beta*R)^(1/sigma) * W, V(W) = (kappa*W)^(1-sigma) /
        # ((1-sigma)(1-rho)), rho = beta*(beta*R)^((1-sigma)/sigma). At an
        # arrival state a', next-period wealth is W' = R*(a' + H).
        rho = beta * (beta * R) ** ((1 - sigma) / sigma)

        def d2_continuation(states, shocks, parameters):
            wealth = R * (states["a"] + H)
            return (kappa * wealth) ** (1 - sigma) / ((1 - sigma) * (1 - rho))

        bp = BellmanPeriod(bm.d2_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 12)}
        # No seeding hint: multi-start covers this box. The midpoint of
        # ``[0, m + H]`` is ~17.7, far above the true optimum (~1.2) and outside
        # L-BFGS-B's basin, so on that seed alone the backup stalls at ~6.97; the
        # clamped ``x0`` candidate lands in the basin and wins on ``res.fun``.
        dr, _, _ = vfi.bellman_step(bp, d2_continuation, grid, scope=cal)
        for a in [1.0, 2.0, 3.0]:
            m = a * R + y
            want = bm.d2_analytical_policy({"a": a}, {}, cal)["c"]
            self.assertAlmostEqual(dr["c"](m), want, delta=self.ATOL)

    def test_d3_single_backup_analytic_continuation(self):
        # D-3 (Blanchard mortality): the hidden Bernoulli survival shock ``live``
        # is a *hidden* shock (c.iset = [m]); its 2 discretization nodes (no
        # ``disc_params`` needed for a discrete shock) are integrated inside the
        # backup. Under the *exact* alive value function -- D-2's closed-form CRRA
        # value with the discount scaled ``beta -> s*beta``, times the survival
        # indicator ``liv'`` (dead => 0) -- a single backup recovers the analytic
        # policy c = kappa_s*(m + H). The block reads the *arrival* ``liv``
        # (utility while alive), so E_live supplies the mortality discount and the
        # Euler FOC is exact: this is the discrete-shock Tier-2 benchmark.
        # Supplying that continuation is what lets this test leave ``liv`` off the
        # grid; iterating instead rebuilds the continuation from the value grid,
        # which must then carry ``liv`` as an axis -- see
        # test_d3_iterated_converges_to_analytic.
        cal = bm.d3_calibration
        beta, R, sigma = cal["DiscFac"], cal["R"], cal["CRRA"]
        s, y = cal["SurvivalProb"], cal["y"]
        H = y / (R - 1)  # human wealth
        beta_eff = s * beta  # mortality as an effective discount (E[liv'] = s*liv)
        kappa_s = (R - (beta_eff * R) ** (1 / sigma)) / R
        # 1 - rho_s == kappa_s (the D-2 identity with beta -> s*beta), so this is
        # exactly D-2's value function at the mortality-adjusted discount.
        rho_s = beta_eff * (beta_eff * R) ** ((1 - sigma) / sigma)

        def d3_continuation(states, shocks, parameters):
            wealth = R * (states["a"] + H)
            v_alive = (kappa_s * wealth) ** (1 - sigma) / ((1 - sigma) * (1 - rho_s))
            return states["liv"] * v_alive  # gated by survival: dead => 0

        bp = BellmanPeriod(bm.d3_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 12)}
        # ``liv`` is an (ungridded) arrival state -> fix the alive slice in scope;
        # the continuation sees the transitioned liv' = live in {0, 1}. This
        # continuation is supplied, not rebuilt from a value grid, so the
        # ungridded-arrival-state guard does not apply. No seeding hint: as in D-2,
        # multi-start covers the [0, m + H] box whose midpoint stalls.
        scope = {**cal, "liv": 1.0}
        dr, _, _ = vfi.bellman_step(bp, d3_continuation, grid, scope=scope)
        for a in [1.0, 2.0, 3.0]:
            m = a * R + y
            want = float(np.asarray(bm.d3_analytical_policy({"a": a}, {}, cal)["c"]))
            self.assertAlmostEqual(dr["c"](m), want, delta=self.ATOL)

    def test_multi_axis_iset_coordinate_is_gathered(self):
        # Gridding case_3 over BOTH a and theta makes m = a + theta vary along two
        # grid axes, so no axis can be relabelled m. The 25 (m, c) pairs are
        # samples of one 1-D rule and are gathered into it: c* = m.
        dr, _, _ = vfi.bellman_step(
            case_3["bp"],
            bp_terminal,
            {"a": np.linspace(0.1, 2.0, 5), "theta": np.linspace(-1, 1, 5)},
            scope={**case_3["calibration"], "psi": 0.0},
        )
        for m in [0.0, 0.5, 1.0, 2.0]:
            self.assertAlmostEqual(dr["c"](m), m, delta=self.ATOL)

    def test_multi_axis_iset_coordinate_needs_a_lone_iset_variable(self):
        # m = a + b varies along both gridded axes, so no axis can be relabelled
        # to it -- and the iset carries a second variable g, so each g slice would
        # gather a different m coordinate. Fail loudly rather than fit one.
        block = DBlock(
            name="two_axis_prestate_plus_iset_var",
            dynamics={
                "m": lambda a, b: a + b,
                "c": Control(["m", "g"], agent="consumer"),
                "a": lambda m, c: m - c,
                "u": lambda c, g: -((c - g) ** 2),
            },
            reward={"u": "consumer"},
        )
        bp = BellmanPeriod(block, "beta", {"beta": 0.9})
        with self.assertRaises(NotImplementedError):
            vfi.bellman_step(
                bp,
                bp_terminal,
                {
                    "a": np.linspace(0.1, 1.0, 4),
                    "b": np.linspace(0.1, 1.0, 4),
                    "g": np.linspace(0.1, 1.0, 3),
                },
            )

    def test_project_to_iset_drops_extra_axis_and_reindexes(self):
        # A grid wider than the iset composes both moves in one pass: the derived
        # variable m claims its source axis a (reindex), and the leftover axis b
        # -- invariant here -- is dropped.
        policy = xr.DataArray(
            np.tile(np.arange(3.0)[:, None], (1, 3)),  # varies along a, flat in b
            dims=["a", "b"],
            coords={"a": [0, 1, 2], "b": [0, 1, 2]},
        )
        m_coord = np.add.outer(2.0 * np.arange(3.0), np.zeros(3))  # m = 2a, flat in b
        out = vfi._project_to_iset(policy, ["a", "b"], ["m"], {"m": m_coord}, "c")
        self.assertEqual(list(out.dims), ["m"])
        self.assertTrue(np.allclose(out["m"].values, [0.0, 2.0, 4.0]))
        self.assertTrue(np.allclose(out.values, [0.0, 1.0, 2.0]))

    def test_project_to_iset_non_monotone_raises(self):
        # A non-monotone grid-axis -> iset-coordinate map would make the
        # reindex-then-interp ill-posed; the monotonicity check fails loudly.
        policy = xr.DataArray(np.zeros(5), dims=["a"], coords={"a": np.arange(5.0)})
        non_monotone = np.array([0.0, 1.0, 0.5, 2.0, 1.5])  # not sorted
        with self.assertRaises(ValueError):
            vfi._project_to_iset(policy, ["a"], ["m"], {"m": non_monotone}, "c")


class test_vfi_solve_bellman(unittest.TestCase):
    """
    Phase-2 design: ``vfi.solve_bellman`` — value-function iteration
    that drives ``bellman_step`` to a fixed point, rebuilding the continuation
    from each iterate's value grid via ``vfi.value_array_to_function``.

    The headline test is **D-4**: a deterministic CRRA model with a binding
    borrowing constraint and impatience, which has *no closed form*. It is
    validated against the package's own independent oracle
    ``d4_vfi_reference_policy`` (a dense cash-on-hand VFI) — a solver-vs-solver
    check that exercises the convergence loop and active-bound handling.
    """

    def test_d4_converges_to_reference_oracle(self):
        # D-4 has no closed form; compare the converged policy to the dense-grid
        # VFI oracle. Two independent exact solvers should agree to ~1% (grid
        # interpolation error), which is well inside a 2% band.
        cal = bm.d4_calibration
        R, y = cal["R"], cal["y"]
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.0, 7.5, 25)}
        dr, value_array, policy_array = vfi.solve_bellman(
            bp, grid, scope=cal, tol=1e-6, max_iter=1000
        )
        # The loop reports convergence via value_array.attrs (O5).
        self.assertTrue(value_array.attrs["converged"])
        self.assertGreater(value_array.attrs["n_iter"], 1)
        self.assertLess(value_array.attrs["residual"], 1e-6)
        # Match the oracle across the binding (low m) and slack (high m) regions.
        for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
            m = a * R + y
            got = dr["c"](m)
            want = float(np.asarray(bm.d4_vfi_reference_policy({"a": a}, {}, cal)["c"]))
            self.assertAlmostEqual(got, want, delta=2e-2)
        # Return contract: gridded value + per-control gridded policy.
        self.assertIsInstance(value_array, xr.DataArray)
        self.assertEqual(list(policy_array["c"].dims), ["a"])

    def test_d2_iterated_converges_to_analytic(self):
        # D-2 (infinite-horizon CRRA, natural borrowing limit c <= m + H,
        # H = y/(R-1) = human wealth): iterate solve_bellman to a fixed point and
        # recover the unconstrained closed form c = kappa*(m + H),
        # kappa = (R - (beta*R)^(1/sigma))/R, at interior states.
        #
        # The artificial_borrowing_constraint flag is required. Without it, from
        # V0 = 0 the first backup has no saving motive and rides the upper bound
        # to c = m + H (a' = -H, the singular "consume all human wealth" point);
        # the self-built continuation is then extrapolated into the V -> -inf
        # wall below the grid and the loop settles on that flat, wrong fixed
        # point (c ~ 35). The flag confines a' to the grid, so the continuation
        # is only interpolated.
        #
        # The grid floor is the slack artificial borrowing limit: a fraction of
        # human wealth H, low enough not to bind at the tested interior states
        # (a' there stays well above -H/3), but not so low that the near-singular
        # deep-borrowing region destabilizes the value iteration.
        #
        # Grid resolution and tol are kept deliberately coarse for speed (the
        # β = 0.96 contraction fixes the iteration count, so a loose tol is the
        # main lever); recovery of the closed form is then good to a few percent.
        cal = bm.d2_calibration
        R, y = cal["R"], cal["y"]
        H = y / (R - 1.0)  # human wealth; natural borrowing limit is a' >= -H
        bp = BellmanPeriod(bm.d2_block, "DiscFac", cal)
        grid = {"a": np.linspace(-H / 3.0, 8.0, 16)}
        dr, value_array, policy_array = vfi.solve_bellman(
            bp,
            grid,
            scope=cal,
            tol=1e-2,
            max_iter=2000,
            artificial_borrowing_constraint=True,
        )
        self.assertTrue(value_array.attrs["converged"])
        for a in [1.0, 2.0, 3.0, 5.0]:
            m = a * R + y
            want = bm.d2_analytical_policy({"a": a}, {}, cal)["c"]
            self.assertAlmostEqual(dr["c"](m), want, delta=5e-2)

    def test_d3_iterated_converges_to_analytic(self):
        # D-3 (Blanchard mortality) iterated to a fixed point, with the survival
        # state liv ON THE GRID. That is what makes the mortality channel visible
        # to the loop: the continuation rebuilt from the value grid is a function
        # of both a' and liv', so E_live[W(a', liv')] = s*W(a', 1) supplies the
        # survival discount and the liv = 1 slice is exactly D-2 at beta -> beta*s,
        # recovering c = kappa_s*(m + H). With liv left off the grid the
        # continuation cannot depend on liv', the 2-node expectation over `live`
        # has a live-free integrand, s cancels, and the loop converges to the
        # no-mortality kappa instead -- which is why value_array_to_function now
        # refuses an ungridded arrival state rather than discarding it.
        #
        # The dead slice is degenerate: at liv = 0 reward and continuation are both
        # 0, so the objective is constant in c and the optimizer returns its seed.
        # bellman_step marks those points UNIDENTIFIED and the iset projection
        # takes its invariance check and its surviving slice over identified points
        # only (liv is outside c's iset = [m], so the axis is dropped).
        cal = bm.d3_calibration
        R, y = cal["R"], cal["y"]
        beta, sigma = cal["DiscFac"], cal["CRRA"]
        H = y / (R - 1.0)
        # Grid floor / coarse tol as in test_d2_iterated_converges_to_analytic: the
        # slack artificial borrowing limit at -H/3 keeps the continuation
        # interpolated, and beta*s = 0.9504 fixes the iteration count.
        bp = BellmanPeriod(bm.d3_block, "DiscFac", cal)
        grid = {"a": np.linspace(-H / 3.0, 8.0, 16), "liv": np.array([0.0, 1.0])}
        dr, value_array, _ = vfi.solve_bellman(
            bp,
            grid,
            scope=cal,
            tol=1e-2,
            max_iter=2000,
            artificial_borrowing_constraint=True,
        )
        self.assertTrue(value_array.attrs["converged"])
        # V(a, 0) = 0 exactly at every iterate (zero reward, zero continuation).
        self.assertEqual(float(np.abs(value_array.sel(liv=0.0)).max()), 0.0)

        # Two assertions: the policy is near kappa_s, and -- the point of the
        # benchmark -- it is nearer kappa_s than the no-mortality kappa, which the
        # solver produced before the survival state entered the grid. The residual
        # is a systematic ~4% under-consumption from the coarse grid and the
        # liquidity-constraint depression at the grid floor (tightening tol alone
        # does not shrink it), the same character as D-2's few-percent recovery.
        kappa = (R - (beta * R) ** (1 / sigma)) / R  # mortality ignored
        for a in [1.0, 2.0, 3.0, 5.0]:
            m = a * R + y
            got = dr["c"](m)
            want = float(np.asarray(bm.d3_analytical_policy({"a": a}, {}, cal)["c"]))
            self.assertAlmostEqual(got, want, delta=8e-2)
            self.assertLess(abs(got - want), abs(got - kappa * (m + H)))

    def test_max_iter_one_matches_bellman_step(self):
        # Iteration 1 uses the terminal (zero) continuation, so max_iter=1 is
        # exactly a single bellman_step under a terminal continuation (loop
        # wiring check). It cannot converge in one step -> converged=False + warn.
        cal = bm.d4_calibration
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 8)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dr_loop, va_loop, _ = vfi.solve_bellman(bp, grid, scope=cal, max_iter=1)
        dr_step, va_step, _ = vfi.bellman_step(bp, bp_terminal, grid, scope=cal)
        self.assertFalse(va_loop.attrs["converged"])
        self.assertEqual(va_loop.attrs["n_iter"], 1)
        self.assertTrue(np.allclose(va_loop.values, va_step.values))
        for a in [1.0, 2.0, 3.0]:
            m = cal["R"] * a + cal["y"]
            self.assertAlmostEqual(dr_loop["c"](m), dr_step["c"](m), delta=1e-6)

    def test_nonconvergence_warns_then_raises(self):
        # A one-iteration run never converges: it warns by default (returning the
        # last iterate, the scipy OptimizeResult.success convention, O5), and
        # raises only when the caller opts in.
        cal = bm.d4_calibration
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 6)}
        with self.assertWarns(UserWarning):
            vfi.solve_bellman(bp, grid, scope=cal, max_iter=1)
        with self.assertRaises(RuntimeError):
            vfi.solve_bellman(
                bp, grid, scope=cal, max_iter=1, raise_on_nonconvergence=True
            )

    def test_u2_iterated_converges_to_analytic(self):
        # U-2 (log utility, normalized, no borrowing constraint): a hidden
        # permanent-income shock psi that is degenerate at sigma_psi = 0 (single
        # node at psi = 1), so the hidden-shock expectation in each backup is
        # exact. Iterate solve_bellman to a fixed point and recover the PIH
        # closed form c = (1 - beta)(m + 1/r) at interior states.
        #
        # Like D-2, U-2 borrows against human wealth h = 1/r, so the iteration
        # rides the control bound without the artificial_borrowing_constraint
        # flag; with it, next-period assets stay on the grid and the continuation
        # is only interpolated. The grid floor is a slack fraction of
        # human wealth: -h/2 sits between the liquidity-depression bias of too
        # high a floor and the deep-borrowing instability of too low one. tol is
        # coarse for speed (the beta = 0.96 contraction fixes the iteration
        # count), giving recovery to a few percent.
        cal = bm.u2_calibration
        R = cal["R"]
        h = 1.0 / (R - 1.0)  # human wealth (normalized); natural limit a >= -h
        bp = BellmanPeriod(bm.u2_block, "DiscFac", cal)
        grid = {"a": np.linspace(-h / 2.0, 8.0, 20)}
        dr, value_array, _ = vfi.solve_bellman(
            bp,
            grid,
            scope=cal,
            tol=1e-2,
            max_iter=2000,
            artificial_borrowing_constraint=True,
        )
        self.assertTrue(value_array.attrs["converged"])
        for a in [1.0, 2.0, 3.0, 5.0]:
            m = R * a + 1.0  # psi = 1 at sigma_psi = 0
            want = float(
                bm.u2_analytical_policy({"a": torch.tensor(float(a))}, {}, cal)["c"]
            )
            self.assertAlmostEqual(dr["c"](m), want, delta=5e-2)

    def test_d1_finite_horizon_converges_to_analytic(self):
        # D-1 (finite-horizon log utility, T = 5): the time-varying rule
        # c_t = (1-beta)/(1-beta^(T-t)) * W, the only benchmark with a
        # non-stationary policy and an integer time axis.
        #
        # The finite horizon needs no special code path, because ``t`` is an
        # ordinary arrival state (``t: lambda t: t + 1``) and the horizon lives in
        # the reward's ``(t < T)`` cutoff. Gridding ``t`` alongside ``W`` makes
        # backward induction an ordinary fixed point in the extended state space:
        # each backup carries information one period further back, so the residual
        # falls to ~0 after about one pass per period and stays there. Hence the
        # iteration count below is O(T) even at tol = 1e-9, against the ~124 that
        # D-2's beta = 0.96 contraction needs at a far looser tol -- the signature
        # of exact propagation rather than geometric convergence.
        #
        # The t axis must extend to T + 1, one slice PAST the last period the
        # reward is nonzero. t' = t + 1 steps off the top of the axis, where the
        # continuation extrapolates linearly (value_array_to_function); with the
        # axis stopping at T, the last two slices are V(.,T-1) = log W and
        # V(.,T) = 0, so the extrapolated "afterlife" at t = T+1 is -log W. The
        # backup then maximizes that, the residual diverges, and the policy
        # collapses. Two identically-zero slices (t = T and t = T+1) make the
        # extrapolation flat at zero, which is the correct terminal condition.
        #
        # W is spaced geometrically: linear interpolation of V = alpha_t + log W
        # has derivative error O(log r) in the grid ratio r, and the Euler equation
        # u'(c) = beta*R*V_W(W') carries that straight into the same relative error
        # in c. Geometric spacing equalizes log r across the grid; r ~ 1.06 here
        # holds the recovery inside ~1.2%.
        cal = bm.d1_calibration
        T = cal["T"]
        bp = BellmanPeriod(bm.d1_block, "DiscFac", cal)
        grid = {"W": np.geomspace(0.02, 8.0, 100), "t": np.arange(0, T + 2)}
        dr, value_array, policy_array = vfi.solve_bellman(
            bp, grid, scope=cal, tol=1e-9, max_iter=30
        )
        self.assertTrue(value_array.attrs["converged"])
        # One pass per period plus one to detect no further change; emphatically
        # not the hundreds a discounted infinite-horizon contraction needs.
        self.assertLessEqual(value_array.attrs["n_iter"], T + 2)
        self.assertGreater(value_array.attrs["n_iter"], 1)

        # Past the horizon the reward is cut off and the continuation is zero, so
        # the value is exactly zero -- the terminal condition, derived not declared.
        self.assertEqual(float(np.abs(value_array.sel(t=T)).max()), 0.0)
        self.assertEqual(float(np.abs(value_array.sel(t=T + 1)).max()), 0.0)

        # The time-varying rule, at every period the agent actually consumes.
        for t in range(T):
            for W in [0.5, 1.0, 2.0, 3.0]:
                got = float(dr["c"](W, t))
                want = float(
                    np.asarray(bm.d1_analytical_policy({"W": W, "t": t}, {}, cal)["c"])
                )
                self.assertAlmostEqual(got / want, 1.0, delta=2e-2)

        # The point of a finite horizon: the MPC rises as the horizon shortens,
        # and in the last consuming period the agent consumes everything.
        for W in [1.0, 2.0]:
            mpcs = [float(dr["c"](W, t)) / W for t in range(T)]
            self.assertTrue(all(np.diff(mpcs) > 0), f"MPC not rising: {mpcs}")
            self.assertAlmostEqual(float(dr["c"](W, T - 1)), W, delta=1e-3)

        self.assertEqual(list(policy_array["c"].dims), ["W", "t"])

    def test_value_array_to_function_interpolates_and_extrapolates(self):
        # The continuation reproduces the grid at the nodes, interpolates between
        # them, and extrapolates linearly past the edges (so an off-grid
        # next-period state during a backup never returns NaN).
        cal = bm.d4_calibration
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        value_array = xr.DataArray(
            np.array([0.0, 1.0, 2.0, 3.0]),  # V = a, slope 1
            dims=["a"],
            coords={"a": [0.0, 1.0, 2.0, 3.0]},
        )
        wf = vfi.value_array_to_function(value_array, bp)
        self.assertAlmostEqual(wf({"a": 1.0}, {}, cal), 1.0)  # node
        self.assertAlmostEqual(wf({"a": 1.5}, {}, cal), 1.5)  # interpolated
        self.assertAlmostEqual(wf({"a": 5.0}, {}, cal), 5.0)  # extrapolated above
        self.assertAlmostEqual(wf({"a": -2.0}, {}, cal), -2.0)  # extrapolated below

    def test_value_array_to_function_integrates_observed_shock_axis(self):
        # An observed-shock axis is integrated out of the arrival value:
        # W(s) = E_obs[V(s, obs)]. Build V(a, theta) over
        # case_1's Normal shock theta on its discretization nodes; the continuation
        # must return the shock-weighted expectation over the theta axis.
        from skagent.distributions import Normal

        disc = Normal(0, 1).discretize(N=5)
        theta_nodes = np.asarray(disc.points)
        a_axis = np.linspace(0.0, 1.0, 4)
        # V(a, theta) = a + theta^2  ->  W(a) = a + E[theta^2] = a + 1.
        V = a_axis[:, None] + theta_nodes[None, :] ** 2
        value_array = xr.DataArray(
            V, dims=["a", "theta"], coords={"a": a_axis, "theta": theta_nodes}
        )
        wf = vfi.value_array_to_function(
            value_array, case_1["bp"], disc_params={"theta": {"N": 5}}
        )
        for a in [0.0, 0.5, 1.0]:
            self.assertAlmostEqual(
                wf({"a": np.float64(a)}, {}, case_1["calibration"]), a + 1.0, delta=1e-9
            )

    def test_value_array_to_function_rejects_ungridded_arrival_state(self):
        # A value grid missing an arrival state cannot represent any dependence on
        # it, so a continuation rebuilt from that grid would silently discard the
        # value it is handed. D-3's survival state liv is the live example: dropping
        # it cancels the mortality discount and yields the no-mortality policy, a
        # plausible number for a different model. Refuse instead.
        cal = bm.d3_calibration
        bp = BellmanPeriod(bm.d3_block, "DiscFac", cal)
        value_array = xr.DataArray(
            np.linspace(0.0, 3.0, 4), dims=["a"], coords={"a": np.linspace(0.0, 3.0, 4)}
        )
        wf = vfi.value_array_to_function(value_array, bp)
        with self.assertRaises(ValueError):
            wf({"a": 1.0, "liv": 1.0}, {}, cal)
        # Only the value actually handed over is an error; an ``a``-only query is
        # still well posed (nothing is being discarded).
        self.assertAlmostEqual(wf({"a": 1.0}, {}, cal), 1.0)

    def test_value_array_to_function_rejects_misaligned_shock_axis(self):
        # The expectation weights are matched to the discretization nodes
        # positionally, so a shock axis whose coordinate is not those nodes must
        # fail loudly rather than mis-weight.
        from skagent.distributions import Normal

        theta_nodes = np.asarray(Normal(0, 1).discretize(N=5).points)
        value_array = xr.DataArray(
            np.zeros((4, 5)),
            dims=["a", "theta"],
            coords={"a": np.linspace(0, 1, 4), "theta": theta_nodes + 0.5},
        )
        with self.assertRaises(ValueError):
            vfi.value_array_to_function(
                value_array, case_1["bp"], disc_params={"theta": {"N": 5}}
            )


class test_vfi_protocol(unittest.TestCase):
    """
    Phase 1 deliverable: a VFI-fitted decision rule is a drop-in for the
    torch-based ``BellmanPeriod`` stack.

    ``vfi.solve`` returns numpy/xarray-space decision rules (positional, in
    information-set order). ``vfi.tensor_decision_rule`` wraps each so it
    accepts and returns torch tensors. The wrapped dict then flows, unmodified,
    through ``BellmanPeriod.compute_controls`` and ``BellmanEquationLoss``.
    """

    def _solve_tensor_dr(self, case, state_grid):
        dr, _, _ = vfi.solve(
            case["block"],
            terminal_continuation,
            state_grid,
            scope=case["calibration"],
        )
        return {c: vfi.tensor_decision_rule(rule) for c, rule in dr.items()}

    def test_compute_controls_roundtrip(self):
        # case_0: u = -c^2, iset = [a] -> c* = 0. The tensorized VFI rule, fed
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
        # zero continuation the VFI policy is exactly optimal, so V(s) = 0 and
        # the Bellman residual u + beta*E[V(s')] - V(s) vanishes. We only assert
        # the loss is finite and (here) ~0 -- the point is that the VFI dr is a
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


class test_vfi_horizon(unittest.TestCase):
    """
    The stopping rule assumes an infinite horizon. Fisher's two-period problem
    is the smallest model with a closed form for which no fixed point is the
    right answer, so it pins what the loop says about a horizon it cannot see.
    """

    def _fisher(self, n=13, a_max=4.0):
        bp = BellmanPeriod(fisher.block, "DiscFac", fisher.calibration)
        avals = np.linspace(0.0, a_max, n)
        m = fisher.calibration["Rfree"] * avals + fisher.calibration["y"]
        exact = fisher.analytical_policy({"m": m}, {}, fisher.calibration)["c"]
        return bp, {"a": avals}, np.asarray(exact)

    def test_finite_horizon_recovers_the_closed_form(self):
        # T backups of the one-period block IS the T-period problem, so
        # max_iter=T is exact up to grid error.
        bp, grid, exact = self._fisher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, policy = vfi.solve_bellman(bp, grid, max_iter=fisher.T)

        c = np.asarray(policy["c"]).ravel()
        self.assertLess(np.abs(c - exact).max(), 5e-2)

    def test_iterating_past_the_horizon_answers_another_question(self):
        # The fixed point exists; it is the infinite-horizon policy, and the
        # loop cannot tell that the caller wanted two periods. Iterating far
        # past T is much further from the closed form than stopping at it.
        bp, grid, exact = self._fisher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, at_horizon = vfi.solve_bellman(bp, grid, max_iter=fisher.T)
            _, _, past_horizon = vfi.solve_bellman(bp, grid, max_iter=60)

        error_at = np.abs(np.asarray(at_horizon["c"]).ravel() - exact).max()
        error_past = np.abs(np.asarray(past_horizon["c"]).ravel() - exact).max()
        self.assertGreater(error_past, 10 * error_at)

    def test_nonconvergence_message_names_both_readings(self):
        # Reaching max_iter is a failure on an infinite-horizon problem and the
        # expected outcome at a finite horizon, and the solver cannot tell
        # which; the message must not assert the first.
        bp, grid, _ = self._fisher()
        with self.assertWarns(UserWarning) as caught:
            vfi.solve_bellman(bp, grid, max_iter=fisher.T)

        message = str(caught.warning)
        self.assertIn("max_iter", message)
        self.assertIn("finite horizon", message)


class test_vfi_stateless(unittest.TestCase):
    """A period with no arrival states is not a dynamic problem."""

    def _one_shot(self):
        # A single decision with no state: x* maximizes -(x - 0.3)^2.
        block = DBlock(
            name="one shot",
            shocks={},
            dynamics={
                "x": Control([], lower_bound=0.0, upper_bound=1.0, agent="a"),
                "u": lambda x: -((x - 0.3) ** 2),
            },
            reward={"u": "a"},
        )
        return BellmanPeriod(block, "beta", {"beta": 0.9})

    def test_bellman_step_accepts_an_empty_grid(self):
        # A contract fact about the grid argument, not a recommendation: the
        # backup degenerates to a single constrained maximization of the reward.
        bp = self._one_shot()
        self.assertEqual(bp.arrival_states, set())

        _, value, policy = vfi.bellman_step(bp, bp_terminal, {})

        self.assertAlmostEqual(float(policy["x"]), 0.3, places=6)
        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_solve_bellman_refuses_a_stateless_period(self):
        # Previously this diverged for all max_iter iterations, or raised from
        # inside the rebuilt continuation with "need at least one array to
        # stack", depending on whether the grid had an axis at all. The message
        # must send the caller to a static solver, not deeper into vfi.
        with self.assertRaises(ValueError) as caught:
            vfi.solve_bellman(self._one_shot(), {})

        message = str(caught.exception)
        self.assertIn("TabularBestResponseSolver", message)
        self.assertIn("solve_multiple_controls", message)
