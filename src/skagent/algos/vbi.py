"""
Value backward induction (VBI).

Derive a decision rule, decision value function, and arrival value function for
a single :class:`~skagent.block.DBlock` stage by backward induction: at each
point of a grid over the decision's information set, solve an exact
:func:`scipy.optimize.minimize` for the control that maximizes the period reward
plus a continuation value.
"""

from skagent.bellman import BellmanPeriod
from skagent.block import DBlock
from inspect import signature
import itertools
import logging
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Mapping, Sequence
import xarray as xr


def get_action_rule(action):
    """
    Build a constant decision rule that ignores its inputs.

    Parameters
    ----------
    action : Any
        The fixed value the rule returns.

    Returns
    -------
    callable
        A zero-argument function ``ar()`` returning ``action``. Used to wrap a
        candidate action as a decision rule during the per-point optimization.
    """

    def ar():
        return action

    return ar


def ar_from_data(da):
    """
    Build a decision rule from a fitted policy ``DataArray``.

    The returned rule follows the library's decision-rule calling convention: it
    takes the control's information-set values as *positional* arguments, in the
    order of ``da.dims`` (which :func:`solve` aligns to ``control.iset``). This
    matches how ``block.transition`` invokes a rule,
    ``dr(*[vals[v] for v in iset])``, so a VBI-fitted rule is a drop-in for the
    rest of the stack.

    Interpolation runs in numpy/xarray space. For a torch-tensor interface, wrap
    the result with :func:`tensor_decision_rule`.

    Parameters
    ----------
    da : xarray.DataArray
        The fitted policy, with one dimension per information-set variable in
        ``control.iset`` order. A zero-dimensional array encodes a constant
        rule (empty information set).

    Returns
    -------
    callable
        A rule ``ar(*args)`` taking one positional argument per dimension of
        *da*. Scalar arguments return a Python scalar; array-like arguments are
        interpolated *pointwise* (not as an outer product) and return a numpy
        array.

    Raises
    ------
    TypeError
        If the number of positional arguments does not match ``da.ndim``.
    """
    dims = list(da.dims)

    def ar(*args):
        if len(args) != len(dims):
            raise TypeError(
                f"decision rule for dims {dims} expects {len(dims)} positional "
                f"argument(s), got {len(args)}"
            )
        if len(dims) == 0:
            # empty information set: a constant rule
            return da.values.tolist()

        batched = any(np.ndim(a) > 0 for a in args)
        if batched:
            # vectorized (pointwise) interpolation: share a single dimension
            # across all coordinate indexers so xarray does not take the outer
            # product of the inputs.
            coords = {
                dim: xr.DataArray(np.asarray(arg).ravel(), dims="_point")
                for dim, arg in zip(dims, args)
            }
            return da.interp(**coords).values

        coords = {dim: arg for dim, arg in zip(dims, args)}
        return da.interp(**coords).values.tolist()

    return ar


def tensor_decision_rule(np_rule, dtype=None, device=None):
    """
    Wrap a numpy-space decision rule so it speaks torch tensors.

    Adapts a rule (e.g. from :func:`ar_from_data`, including the rules returned
    by :func:`solve`) for interop with the torch solving stack
    (``BellmanPeriod`` / ``loss`` / ``solver``). Inputs may be torch tensors or
    numpy/scalars; outputs are torch tensors.

    Because interpolation runs in numpy, the autograd graph is *severed*: the
    returned controls are detached. This rule is therefore valid as a fixed /
    ground-truth / warm-start policy (e.g. an ``other_dr`` or in a value-residual
    loss), but not as a trainable policy in a loss that differentiates through
    the control (FOC-weighted Bellman, Euler).

    Parameters
    ----------
    np_rule : callable
        A numpy-space decision rule taking positional information-set arguments
        and returning numpy/scalar values.
    dtype : torch.dtype, optional
        Output tensor dtype. Defaults to ``torch.float32`` to match grid tensors
        (``grid.py`` builds them with ``torch.FloatTensor``).
    device : torch.device, optional
        Output tensor device. Defaults to the stack's device
        (``skagent.grid.device``).

    Returns
    -------
    callable
        A rule ``tdr(*args)`` accepting torch tensors (or numpy/scalars) and
        returning a detached torch tensor on *device* with dtype *dtype*.
    """
    import torch
    from skagent.grid import device as default_device

    dtype = torch.float32 if dtype is None else dtype
    device = default_device if device is None else device

    def tdr(*args):
        np_args = [a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args]
        out = np_rule(*np_args)
        return torch.as_tensor(np.asarray(out), dtype=dtype, device=device)

    return tdr


Grid = Mapping[str, Sequence]


def grid_to_data_array(
    grid: Grid = {},  ## TODO: Better data structure here.
):
    """
    Construct a zero-valued ``DataArray`` over the coordinates of a grid.

    Parameters
    ----------
    grid : Grid, optional
        A mapping from variable labels to a sequence of numerical values. An
        empty mapping yields a zero-dimensional array.

    Returns
    -------
    xarray.DataArray
        An array whose dimensions and coordinates are those of *grid*.
    """

    coords = {**grid}

    da = xr.DataArray(
        np.empty([len(v) for v in coords.values()]), dims=coords.keys(), coords=coords
    )

    return da


def solve(block: DBlock, continuation, state_grid: Grid, disc_params={}, scope={}):
    """
    Solve a ``DBlock`` stage by backward induction on the value function.

    At each point of *state_grid*, the optimal control(s) are found with
    :func:`scipy.optimize.minimize`, maximizing the period reward plus the
    *continuation* value of the resulting states. The tabulated optima are then
    interpolated into a decision rule.

    VBI assumes *full observation*: the decision conditions on its complete
    information set and the per-point optimization never integrates over
    unobserved variables. (The only expectation machinery in this module is in
    ``block.get_arrival_value_function``; the optimization here does not use it.)
    Hidden-shock problems whose optimum requires an expectation are out of scope.

    Parameters
    ----------
    block : DBlock
        The stage to solve. Must contain at most one control variable;
        multi-control stages raise ``Exception``.
    continuation : callable
        The continuation value function, called with the post-transition values
        of the variables named in its signature. Fold any discount factor into
        this function (the backup is ``reward + continuation``).
    state_grid : Grid
        A grid over the control's information set: one axis per variable the
        decision may condition on. The returned decision rule takes these as
        positional arguments in ``control.iset`` order. Variables the dynamics
        need but the decision does not (e.g. a shock that only enters the
        transition) go in *scope*, not here. For an empty information set, pass
        ``{}``.
    disc_params : Mapping, optional
        Discretization parameters for the shock distribution, forwarded to
        ``block.get_arrival_value_function``.
    scope : Mapping, optional
        The fixed scope for the per-point optimization: merged with each grid
        point to form the ``pre_states`` under which the dynamics, reward, and
        continuation are evaluated.

        .. note::
           This is broader than ``calibration`` elsewhere in the library, which
           denotes fixed, single-valued *parameters* only. Here (legacy VBI
           usage) it is a general scope bag that also holds fixed exogenous
           values outside the information set, such as a shock realization
           ``psi``. Read it as "scope," not "parameters."

    Returns
    -------
    dr_from_data : dict of callable
        One decision rule per control, keyed by control symbol; each takes its
        information-set values as positional arguments in ``control.iset`` order.
    dec_vf : callable
        The decision value function for the fitted rule.
    arr_vf : callable
        The arrival value function for the fitted rule (takes the shock
        expectation via *disc_params*).
    """

    # state-rule value function
    srv_function = block.get_state_rule_value_function_from_continuation(
        continuation, screen=True
    )

    # get_controls() returns a dict[sym, Control]; VBI works with the
    # ordered list of control symbols.
    controls = list(block.get_controls())

    # pseudo
    policy_array = grid_to_data_array(state_grid)
    value_array = grid_to_data_array(state_grid)

    # loop through every point in the state grid
    for state_point in itertools.product(*state_grid.values()):
        # build a dictionary from these states, as scope for the optimization
        state_vals = {k: v for k, v in zip(state_grid.keys(), state_point)}

        # The value of the action is computed given the fixed scope and the
        # states for the current point on the state-grid.
        pre_states = scope.copy()
        pre_states.update(state_vals)

        # prepare function to optimize
        def negated_value(a):
            dr = {c: get_action_rule(a[i]) for i, c in enumerate(controls)}

            # negative, for minimization later
            return -srv_function(pre_states, dr)

        if len(controls) == 0:
            # if no controls, no optimization is necessary
            pass
        elif len(controls) == 1:
            ## get lower bound.
            ## assumes only one control currently
            lower_bound = -1e12  # a very low number
            feq = block.dynamics[controls[0]].lower_bound
            if feq is not None:
                lower_bound = feq(
                    *[pre_states[var] for var in signature(feq).parameters]
                )

            ## get upper bound
            ## assumes only one control currently
            upper_bound = 1e12  # a very high number
            feq = block.dynamics[controls[0]].upper_bound

            if feq is not None:
                upper_bound = feq(
                    *[pre_states[var] for var in signature(feq).parameters]
                )

            bounds = ((lower_bound, upper_bound),)

            res = minimize(  # choice of
                negated_value,
                1,  # x0 is starting guess, here arbitrary.
                bounds=bounds,
            )

            dr_best = {c: get_action_rule(res.x[i]) for i, c in enumerate(controls)}

            if res.success:
                policy_array.sel(**state_vals).variable.data.put(
                    0, res.x[0]
                )  # will only work for scalar actions
                value_array.sel(**state_vals).variable.data.put(
                    0, srv_function(pre_states, dr_best)
                )
            else:
                print(f"Optimization failure at {state_vals}.")
                print(res)

                dr_best = {c: get_action_rule(res.x[i]) for i, c in enumerate(controls)}

                policy_array.sel(**state_vals).variable.data.put(0, res.x[0])  # ?
                value_array.sel(**state_vals).variable.data.put(
                    0, srv_function(pre_states, dr_best)
                )
        elif len(controls) > 1:
            raise Exception(
                f"Value backup iteration is not yet implemented for stages with {len(controls)} > 1 control variables."
            )

    # Use the xarray interpolator to create a decision rule. state_grid is the
    # control's information set (see the docstring); transposing to iset order
    # makes solve own the contract that the rule's positional arguments follow
    # control.iset, regardless of how the caller ordered the grid.
    dr_from_data = {
        c: ar_from_data(policy_array.transpose(*block.dynamics[c].iset))
        for c in controls
    }

    dec_vf = block.get_decision_value_function(dr_from_data, continuation)
    arr_vf = block.get_arrival_value_function(disc_params, dr_from_data, continuation)

    return dr_from_data, dec_vf, arr_vf


# Sentinels for an "open" (effectively infinite) bound, symmetric with legacy
# ``solve``. A bound this large is treated as absent when seeding ``x0``.
_LOWER_OPEN = -1e12
_UPPER_OPEN = 1e12

# Tolerance for the Mechanism-B single-axis-dependence check (§5): an iset
# coordinate is treated as varying along a grid axis when its spread along that
# axis exceeds this. Flat absolute, matching the test ATOL scale; hardening
# (parameterize + relative/absolute combo) is deferred to Phase 3 (i66_todo.md).
_MECHB_TOL = 1e-3


def _project_to_iset(
    policy_buf: np.ndarray,
    grid_axes: Sequence[str],
    iset: Sequence[str],
    iset_coords: Mapping[str, np.ndarray],
    control_sym: str,
) -> xr.DataArray:
    """
    Reindex a grid-tabulated policy onto a control's information set when an iset
    variable is a *derived pre-state* rather than a grid axis (Mechanism B, §5).

    Legacy :func:`solve` gridded directly over the iset; the ``BellmanPeriod``
    backup grids over arrival states (and observed shocks) — the value
    function's domain — so a control whose iset is a derived pre-state (e.g.
    ``m = a·R + y``) needs its policy re-expressed on that coordinate. Each iset
    variable is required to map from **exactly one** grid axis by a strictly
    monotone, otherwise-invariant relation (the 1-axis-per-iset-var case of O3);
    the policy axes are then relabeled to the iset variables and re-coordinated
    onto their computed values, so :func:`ar_from_data` can interpolate over the
    regular iset coordinate.

    Parameters
    ----------
    policy_buf : numpy.ndarray
        The optimal control tabulated over the state grid (shape = grid shape,
        axes in *grid_axes* order).
    grid_axes : sequence of str
        The state-grid axis names, in array order.
    iset : sequence of str
        The control's information set (the target coordinates).
    iset_coords : Mapping[str, numpy.ndarray]
        The iset-variable values computed over the grid (each shape = grid
        shape), from :meth:`BellmanPeriod.compute_pre_state` at each point.
    control_sym : str
        The control symbol, for error messages.

    Returns
    -------
    xarray.DataArray
        The policy indexed by the iset coordinates, transposed into *iset*
        order, ready for :func:`ar_from_data`.

    Raises
    ------
    NotImplementedError
        If the grid and iset have different rank (a grid wider than the iset is
        a Mechanism-A *reduction*, design §9 step 3), or if an iset variable
        depends on more than one grid axis (general scattered reindexing, O3).
    ValueError
        If the grid-axis → iset-coordinate map is not strictly monotone, so the
        reindex-then-``interp`` would be ill-posed (§5 monotonicity assert).
    """
    if len(iset) != len(grid_axes):
        raise NotImplementedError(
            f"Projecting control '{control_sym}'s policy from grid axes "
            f"{list(grid_axes)} onto information set {list(iset)} needs a "
            "same-rank reindex; a grid wider than the iset is a Mechanism-A "
            "reduction, which is out of scope here (design §9 step 3)."
        )

    iset_var_at_axis: dict[int, str] = {}  # grid-axis index -> iset variable
    coord_at_axis: dict[int, np.ndarray] = {}  # grid-axis index -> 1-D coord
    for iv in iset:
        coord = np.asarray(iset_coords[iv])
        # Which grid axes does this iset coordinate actually vary along?
        varying = [
            k
            for k in range(coord.ndim)
            if float(np.abs(coord.max(axis=k) - coord.min(axis=k)).max()) > _MECHB_TOL
        ]
        if len(varying) != 1:
            raise NotImplementedError(
                f"Information-set variable '{iv}' of control '{control_sym}' "
                f"varies along grid axes {[grid_axes[k] for k in varying]} "
                "(expected exactly one). General multi-axis scattered "
                "reindexing is out of scope (design §5, O3)."
            )
        k = varying[0]
        if k in iset_var_at_axis:
            raise NotImplementedError(
                f"Grid axis '{grid_axes[k]}' feeds both information-set "
                f"variables '{iset_var_at_axis[k]}' and '{iv}' of control "
                f"'{control_sym}'; Mechanism B requires a one-to-one "
                "grid-axis -> iset-variable map (design §5, O3)."
            )
        # Collapse to the coordinate's 1-D profile along its source axis
        # (it is invariant along every other axis, just checked).
        line = coord[tuple(slice(None) if j == k else 0 for j in range(coord.ndim))]
        diffs = np.diff(line)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise ValueError(
                f"The map from grid axis '{grid_axes[k]}' to information-set "
                f"variable '{iv}' of control '{control_sym}' is not strictly "
                "monotone, so the reindex-then-interp would be ill-posed "
                "(design §5 monotonicity assert)."
            )
        iset_var_at_axis[k] = iv
        coord_at_axis[k] = line

    dims = [iset_var_at_axis[k] for k in range(len(grid_axes))]
    coords = {iset_var_at_axis[k]: coord_at_axis[k] for k in range(len(grid_axes))}
    return xr.DataArray(policy_buf, dims=dims, coords=coords).transpose(*iset)


def bellman_step(
    bp: BellmanPeriod,
    continuation_vf: Callable,
    state_grid: Grid,
    *,
    agent: str | None = None,
    scope: Mapping = {},
    disc_params: Mapping = {},
    x0: float = 1.0,
    x0_policy: Mapping[str, xr.DataArray] | None = None,
) -> tuple[dict[str, Callable], xr.DataArray, dict[str, xr.DataArray]]:
    """
    One exact value backup over *state_grid* on the ``BellmanPeriod`` protocol.

    This is the per-iteration update of value-function iteration: at each grid
    point the optimal control is found with :func:`scipy.optimize.minimize`,
    maximizing the period reward plus the discounted *continuation* value of the
    resulting arrival states. Under a terminal (zero) continuation,
    ``continuation_vf = lambda s, sh, p: 0.0``, the result is the single-step
    solution; the value-iteration wrapper ``solve_bellman`` (§3) iterates it to a
    fixed point.

    Unlike legacy :func:`solve` (which rides the ``DBlock`` continuation API and
    folds the discount factor into the continuation), this speaks the
    ``BellmanPeriod`` protocol the rest of the torch stack uses, with an explicit
    discount factor and multi-reward summation, and is empty-shock-safe.

    .. note::
       This is the **current scope** of the §2 design: a single control, with
       the policy projected onto the control's information set either by the
       grid-equals-iset transpose or, when an iset variable is a derived
       pre-state, by the Mechanism-B reindex (§5); and no internal shock
       discretization (§4). Hidden shocks must be supplied as fixed
       realizations via *scope*; multi-control, ``disc_params``, and a grid
       *wider* than the iset (Mechanism-A reduction) raise
       :class:`NotImplementedError`.

    Parameters
    ----------
    bp : BellmanPeriod
        The recurring period providing the model mechanics.
    continuation_vf : callable
        The continuation value function, called ``continuation_vf(states, shocks,
        parameters)`` on the next-period arrival states (the ``bp.compute_value``
        convention). Terminal continuation is ``lambda s, sh, p: 0.0``.
    state_grid : Grid
        A grid over the control's information set: one axis per variable the
        decision conditions on (arrival states and/or observed shocks). For an
        empty information set, pass ``{}``.
    agent : str, optional
        If given, the period reward sums only this agent's reward symbols.
    scope : Mapping, optional
        Fixed non-shock exogenous values merged into the model parameters. This
        is also where a hidden shock's fixed realization is currently supplied
        (pending internal discretization, §4).
    disc_params : Mapping, optional
        Reserved for internal shock discretization (§4); must be empty for now.
    x0 : float, optional
        Fallback optimizer seed used when a control has an open bound.
    x0_policy : Mapping[str, DataArray], optional
        Warm-start seeds keyed by control symbol (e.g. a previous iterate's
        ``policy_array``); when given, the seed at each grid point is read from
        here. Supplied by :func:`solve_bellman`.

    Returns
    -------
    dr_from_data : dict of callable
        One decision rule per control, keyed by control symbol; each takes its
        information-set values as positional arguments in ``control.iset`` order.
    value_array : xarray.DataArray
        The gridded optimized decision value over the state grid.
    policy_array : dict of xarray.DataArray
        The gridded optimal control(s) over the state grid, keyed by control
        symbol (a dict for forward-compatibility with multi-control, O1).
    """
    controls = list(bp.get_controls())
    if len(controls) != 1:
        raise NotImplementedError(
            f"bellman_step handles exactly one control; got "
            f"{len(controls)} {controls}. Multi-control vectorization is not yet "
            "implemented (design §9 step 3)."
        )
    if disc_params:
        raise NotImplementedError(
            "disc_params (internal shock discretization) is not yet implemented "
            "(design §9 step 6); for now supply any hidden-shock realization "
            "as a fixed value via `scope`."
        )

    control_sym = controls[0]
    iset = bp.block.dynamics[control_sym].iset
    grid_axes = list(state_grid.keys())
    shock_syms = set(bp.get_shocks())
    arrival = bp.arrival_states

    # The state grid is the value function's domain (arrival states + observed
    # shocks); the policy is projected onto the control's iset afterwards. When
    # the grid equals the iset that projection is a transpose; when an
    # iset variable is a derived pre-state it is the Mechanism-B reindex (§5),
    # which also owns the rank/monotonicity guards. A grid *wider* than the
    # iset (Mechanism-A reduction) still raises there (design §9 step 3).
    grid_equals_iset = set(grid_axes) == set(iset)

    params = {**bp.calibration, **scope}
    for s in shock_syms - set(grid_axes):
        if s not in params:
            raise NotImplementedError(
                f"Shock '{s}' is hidden (not in control '{control_sym}'s "
                f"information set) and has no fixed value. For now, supply a "
                "realization via `scope`; integration over hidden shocks is not "
                "yet implemented (design §9 step 6)."
            )

    reward_syms = bp.get_reward_syms(agent)
    lower_func = bp.block.dynamics[control_sym].lower_bound
    upper_func = bp.block.dynamics[control_sym].upper_bound

    shape = tuple(len(state_grid[k]) for k in grid_axes)
    policy_buf = np.empty(shape)
    value_buf = np.empty(shape)
    # When the grid is not the iset, the policy must be reindexed onto the
    # iset's (possibly derived) coordinates; tabulate them alongside the policy.
    iset_coord_buf = (
        {iv: np.empty(shape) for iv in iset} if not grid_equals_iset else None
    )

    for idx in np.ndindex(*shape):
        point_vals = {k: state_grid[k][i] for k, i in zip(grid_axes, idx)}
        states = {k: v for k, v in point_vals.items() if k in arrival}
        obs = {k: v for k, v in point_vals.items() if k in shock_syms}

        # Per-control bounds, evaluated at this point (pre-state available).
        pre = bp.compute_pre_state(control_sym, states, shocks=obs, parameters=params)
        if iset_coord_buf is not None:
            for iv in iset:
                iset_coord_buf[iv][idx] = pre[iv]
        bag = {**params, **obs, **states, **pre}
        lb = (
            lower_func(*[bag[v] for v in signature(lower_func).parameters])
            if lower_func is not None
            else _LOWER_OPEN
        )
        ub = (
            upper_func(*[bag[v] for v in signature(upper_func).parameters])
            if upper_func is not None
            else _UPPER_OPEN
        )

        def negated_value(a):
            ctrl = {control_sym: a[0]}
            rewards = bp.reward_function(
                states, ctrl, shocks=obs, parameters=params, agent=agent
            )
            r = sum(rewards[s] for s in reward_syms)
            post = bp.post_function(states, ctrl, shocks=obs, parameters=params)
            beta = bp.resolve_discount_factor(post)
            s_next = bp.transition_function(states, ctrl, shocks=obs, parameters=params)
            return -(r + beta * continuation_vf(s_next, obs, params))

        # Seed: warm-start > midpoint of finite bounds > x0 fallback (§2).
        if x0_policy is not None:
            seed = float(x0_policy[control_sym].values[idx])
        elif lower_func is not None and upper_func is not None:
            seed = (lb + ub) / 2
        else:
            seed = x0

        res = minimize(negated_value, [seed], bounds=[(lb, ub)])
        if not res.success:
            logging.warning(
                "bellman_step optimization did not converge at %s: %s",
                point_vals,
                res.message,
            )
        policy_buf[idx] = res.x[0]
        value_buf[idx] = -res.fun

    coords = {k: state_grid[k] for k in grid_axes}
    policy_da = xr.DataArray(policy_buf, dims=grid_axes, coords=coords)
    value_array = xr.DataArray(value_buf, dims=grid_axes, coords=coords)

    # Project the gridded policy onto the control's iset: a transpose when the
    # grid is the iset, else the Mechanism-B reindex (§5). Only the
    # decision rule moves to iset coordinates; policy_array stays over the state
    # grid so solve_bellman can warm-start the next iterate at the same points.
    if grid_equals_iset:
        policy_on_iset = policy_da.transpose(*iset)
    else:
        policy_on_iset = _project_to_iset(
            policy_buf, grid_axes, iset, iset_coord_buf, control_sym
        )
    dr_from_data = {control_sym: ar_from_data(policy_on_iset)}
    policy_array = {control_sym: policy_da}

    return dr_from_data, value_array, policy_array
