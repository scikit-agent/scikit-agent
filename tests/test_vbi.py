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
    case_11,
)
import skagent.algos.vbi as vbi
from skagent.bellman import BellmanPeriod
from skagent.distributions import Bernoulli
from skagent.block import Control, DBlock
from skagent.loss import BellmanEquationLoss
from skagent.grid import device
import skagent.models.benchmarks as bm
import skagent.models.consumer as cons
import numpy as np
import xarray as xr
import torch
import unittest
import warnings


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
      construction and does not affect the policy solve (VBI is full-observation
      and never integrates over shocks in its per-point optimization).
    """

    # tolerance on the recovered policy; the optima here are all linear, so
    # grid interpolation is exact and scipy's optimizer is the only error source
    ATOL = 1e-3

    def test_case_0_interior_optimum(self):
        # u = -c^2, unconstrained -> c* = 0 for all a; V(a) = 0.
        state_grid = {"a": np.linspace(0, 2, 11)}
        dr, dec_vf, _ = vbi.solve(
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
        dr, dec_vf, _ = vbi.solve(
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
        dr, _, _ = vbi.solve(
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
        case_5["block"].construct_shocks(case_5["calibration"])
        dr, _, arr_vf = vbi.solve(
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
        case_6["block"].construct_shocks(case_6["calibration"])
        dr, _, arr_vf = vbi.solve(
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
        case_7["block"].construct_shocks(case_7["calibration"])
        dr, _, arr_vf = vbi.solve(
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
        case_8["block"].construct_shocks(case_8["calibration"])
        dr, _, arr_vf = vbi.solve(
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
        dr, dec_vf, _ = vbi.solve(
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


class test_vbi_bellman_step(unittest.TestCase):
    """
    Phase-2 design (§9 steps 1-3): ``vbi.bellman_step`` — one exact value backup
    on the ``BellmanPeriod`` protocol; single- and multi-control (one joint
    ``scipy.minimize`` over the stacked control vector, per-control iset
    projection).

    Under a terminal (zero) continuation each conftest case reduces to a single
    backward-induction step whose optimum is the case's analytic ``optimal_dr``.
    These mirror ``test_vbi_conftest`` (which exercises legacy ``solve``) but
    drive ``bellman_step`` and assert its 3-tuple return contract. The
    Mechanism-B reindex (§5) is also exercised: a control whose information set
    is a derived pre-state (``case_3``'s ``m = a + theta``, D-2's ``m = a·R + y``).
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
        # A real continuation_vf drives the optimum (design §9 step 4). The
        # period reward u = -(a - b)^2 is over the ARRIVAL states (a, b) and is
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
        dr, value_array, policy = vbi.bellman_step(
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
            vbi._project_to_iset(policy, ["a"], [], {}, "c")

    def test_disc_params_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vbi.bellman_step(
                case_0["bp"],
                bp_terminal,
                {"a": np.linspace(0, 2, 5)},
                disc_params={"theta": {"N": 3}},
            )

    # --- iset is a derived pre-state: reproject onto its coordinate (§5) ----

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
        # non-trivial continuation, decoupled from the iteration loop (§3).
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
        # Seed the per-point optimizer near the (modest) optimum. With the
        # consumption floor restored, ``c``'s bounds are both finite, so the
        # default seed is the midpoint of ``[0, m + H]`` (~17.7) — far above the
        # true optimum (~1.2) and outside L-BFGS-B's basin here, so it stalls.
        # A flat warm-start reproduces the robust pre-floor seeding. The general
        # fix (multi-start) is deferred to design.md §8.
        x0_policy = {
            "c": xr.DataArray(np.ones(grid["a"].size), dims=["a"], coords=grid)
        }
        dr, _, _ = vbi.bellman_step(
            bp, d2_continuation, grid, scope=cal, x0_policy=x0_policy
        )
        for a in [1.0, 2.0, 3.0]:
            m = a * R + y
            want = bm.d2_analytical_policy({"a": a}, {}, cal)["c"]
            self.assertAlmostEqual(dr["c"](m), want, delta=self.ATOL)

    def test_mechanism_b_multi_axis_not_implemented(self):
        # Gridding case_3 over BOTH a and theta makes m = a + theta vary along
        # two grid axes -> general scattered reindexing, out of scope in v1
        # (design §5, O3): fail loudly rather than interpolate wrongly.
        with self.assertRaises(NotImplementedError):
            vbi.bellman_step(
                case_3["bp"],
                bp_terminal,
                {"a": np.linspace(0.1, 2.0, 5), "theta": np.linspace(-1, 1, 5)},
                scope={**case_3["calibration"], "psi": 0.0},
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
        out = vbi._project_to_iset(policy, ["a", "b"], ["m"], {"m": m_coord}, "c")
        self.assertEqual(list(out.dims), ["m"])
        self.assertTrue(np.allclose(out["m"].values, [0.0, 2.0, 4.0]))
        self.assertTrue(np.allclose(out.values, [0.0, 1.0, 2.0]))

    def test_project_to_iset_non_monotone_raises(self):
        # A non-monotone grid-axis -> iset-coordinate map would make the
        # reindex-then-interp ill-posed; the monotonicity check fails loudly.
        policy = xr.DataArray(np.zeros(5), dims=["a"], coords={"a": np.arange(5.0)})
        non_monotone = np.array([0.0, 1.0, 0.5, 2.0, 1.5])  # not sorted
        with self.assertRaises(ValueError):
            vbi._project_to_iset(policy, ["a"], ["m"], {"m": non_monotone}, "c")


class test_vbi_solve_bellman(unittest.TestCase):
    """
    Phase-2 design (§9 step 5): ``vbi.solve_bellman`` — value-function iteration
    that drives ``bellman_step`` to a fixed point, rebuilding the continuation
    from each iterate's value grid via ``vbi.value_array_to_function``.

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
        dr, value_array, policy_array = vbi.solve_bellman(
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

    def test_max_iter_one_matches_bellman_step(self):
        # Iteration 1 uses the terminal (zero) continuation, so max_iter=1 is
        # exactly a single bellman_step under a terminal continuation (loop
        # wiring check). It cannot converge in one step -> converged=False + warn.
        cal = bm.d4_calibration
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        grid = {"a": np.linspace(0.5, 5.0, 8)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dr_loop, va_loop, _ = vbi.solve_bellman(bp, grid, scope=cal, max_iter=1)
        dr_step, va_step, _ = vbi.bellman_step(bp, bp_terminal, grid, scope=cal)
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
            vbi.solve_bellman(bp, grid, scope=cal, max_iter=1)
        with self.assertRaises(RuntimeError):
            vbi.solve_bellman(
                bp, grid, scope=cal, max_iter=1, raise_on_nonconvergence=True
            )

    def test_disc_params_not_implemented(self):
        # Internal shock discretization is a later PR (design §9 step 6).
        cal = bm.d4_calibration
        bp = BellmanPeriod(bm.d4_block, "DiscFac", cal)
        with self.assertRaises(NotImplementedError):
            vbi.solve_bellman(
                bp,
                {"a": np.linspace(0.5, 5.0, 6)},
                scope=cal,
                disc_params={"x": {"N": 3}},
            )

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
        wf = vbi.value_array_to_function(value_array, bp)
        self.assertAlmostEqual(wf({"a": 1.0}, {}, cal), 1.0)  # node
        self.assertAlmostEqual(wf({"a": 1.5}, {}, cal), 1.5)  # interpolated
        self.assertAlmostEqual(wf({"a": 5.0}, {}, cal), 5.0)  # extrapolated above
        self.assertAlmostEqual(wf({"a": -2.0}, {}, cal), -2.0)  # extrapolated below

    def test_value_array_to_function_rejects_shock_axis(self):
        # Integrating an observed-shock axis out of the arrival value is a later
        # PR (design §9 step 6); a shock-valued grid axis must fail loudly.
        cal = bm.u2_calibration
        bp = BellmanPeriod(bm.u2_block, "DiscFac", cal)  # has shock 'psi'
        value_array = xr.DataArray(
            np.zeros((2, 2)),
            dims=["a", "psi"],
            coords={"a": [0.0, 1.0], "psi": [0.9, 1.1]},
        )
        with self.assertRaises(NotImplementedError):
            vbi.value_array_to_function(value_array, bp)


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
