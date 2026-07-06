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
import warnings
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

# A coordinate is treated as varying along a grid axis when its spread there
# exceeds this; below it, the axis is considered flat (invariant). Flat absolute
# at the test ATOL scale; relative/absolute hardening is deferred to Phase 3.
_PROJ_TOL = 1e-3


def _project_to_iset(
    policy_da: xr.DataArray,
    grid_axes: Sequence[str],
    iset: Sequence[str],
    iset_coords: Mapping[str, np.ndarray],
    control_sym: str,
) -> xr.DataArray:
    """
    Re-express a grid-tabulated policy as a function of a control's information
    set, ready for :func:`ar_from_data` (design §5).

    The backup grids over the value function's domain (arrival states + observed
    shocks); a control conditions on its own ``iset``. Each iset variable must
    track exactly one grid axis: either it *is* that axis, or it is a derived
    pre-state (e.g. ``m = a·R + y``) that varies, strictly monotonically, only
    along it — in which case that axis is relabeled to the variable and
    recoordinated onto its computed values. Any leftover grid axis is outside the
    iset, so the optimum must be invariant along it; it is dropped (``isel`` 0).

    Raises ``NotImplementedError`` if an iset variable varies along more than one
    grid axis, or two variables share an axis (general scattered reindexing, out
    of scope); ``ValueError`` if a derived map is non-monotone or a dropped axis
    is non-invariant (the reprojection would be ill-posed).
    """
    da = policy_da
    claimed: dict[str, str] = {}  # grid axis -> iset variable it represents
    derived_coords: dict[str, np.ndarray] = {}

    for iv in iset:
        if iv in grid_axes:
            axis = iv  # the variable is itself a grid axis
        else:
            coord = np.asarray(iset_coords[iv])
            varying = [
                k
                for k in range(coord.ndim)
                if float(np.abs(coord.max(axis=k) - coord.min(axis=k)).max())
                > _PROJ_TOL
            ]
            if len(varying) != 1:
                raise NotImplementedError(
                    f"Information-set variable '{iv}' of control '{control_sym}' "
                    f"varies along grid axes {[grid_axes[k] for k in varying]} "
                    "(expected exactly one); scattered reindexing is out of scope."
                )
            (k,) = varying
            axis = grid_axes[k]
            # 1-D profile along the source axis (flat along all others, checked).
            line = coord[tuple(slice(None) if j == k else 0 for j in range(coord.ndim))]
            if not (np.all(np.diff(line) > 0) or np.all(np.diff(line) < 0)):
                raise ValueError(
                    f"The map from grid axis '{axis}' to information-set variable "
                    f"'{iv}' of control '{control_sym}' is not strictly monotone."
                )
            derived_coords[iv] = line
        if axis in claimed:
            raise NotImplementedError(
                f"Grid axis '{axis}' feeds two information-set variables "
                f"('{claimed[axis]}', '{iv}') of control '{control_sym}'; "
                "a one-to-one axis -> variable map is required."
            )
        claimed[axis] = iv

    for ax in grid_axes:
        if ax not in claimed:
            spread = float(np.abs(da.max(dim=ax) - da.min(dim=ax)).max())
            if spread > _PROJ_TOL:
                raise ValueError(
                    f"Control '{control_sym}'s optimum varies along grid axis "
                    f"'{ax}' (spread {spread:.3g}), which is outside its "
                    f"information set {list(iset)}."
                )
            da = da.isel({ax: 0})

    da = da.rename({ax: iv for ax, iv in claimed.items() if ax != iv})
    return da.assign_coords(derived_coords).transpose(*iset)


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
       Current scope of the §2 design: one or more controls, jointly optimized
       by a single :func:`scipy.optimize.minimize` over the stacked control
       vector with per-control bounds, each policy then reprojected onto its own
       information set (:func:`_project_to_iset`, §5). There is no internal shock
       discretization (§4): hidden shocks must be supplied as fixed realizations
       via *scope*, and ``disc_params`` raises :class:`NotImplementedError`.

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
    if len(controls) == 0:
        raise NotImplementedError(
            "bellman_step needs at least one control; a control-free block has "
            "no decision to optimize."
        )
    if disc_params:
        raise NotImplementedError(
            "disc_params (internal shock discretization) is not yet implemented "
            "(design §9 step 6); for now supply any hidden-shock realization "
            "as a fixed value via `scope`."
        )

    grid_axes = list(state_grid.keys())
    shock_syms = set(bp.get_shocks())
    arrival = bp.arrival_states

    # Per-control static info: controls may have *different* information sets and
    # bounds, so everything per-control is keyed by control symbol. The state
    # grid is the value function's domain (arrival states + observed shocks),
    # shared across controls; each control's policy is projected onto its own
    # iset afterwards (§5).
    iset_by_control = {c: bp.block.dynamics[c].iset for c in controls}
    lower_by_control = {c: bp.block.dynamics[c].lower_bound for c in controls}
    upper_by_control = {c: bp.block.dynamics[c].upper_bound for c in controls}

    params = {**bp.calibration, **scope}
    iset_union = {iv for c in controls for iv in iset_by_control[c]}
    for s in shock_syms - set(grid_axes):
        if s not in params:
            raise NotImplementedError(
                f"Shock '{s}' is hidden (in no control's information set "
                f"{sorted(iset_union)}) and has no fixed value. For now, supply "
                "a realization via `scope`; integration over hidden shocks is "
                "not yet implemented (design §9 step 6)."
            )

    reward_syms = bp.get_reward_syms(agent)

    shape = tuple(len(state_grid[k]) for k in grid_axes)
    value_buf = np.empty(shape)
    policy_buf = {c: np.empty(shape) for c in controls}
    # Tabulate each control's information-set coordinates over the grid, in case
    # a variable is a derived pre-state that must be reprojected (§5).
    iset_coord_buf = {
        c: {iv: np.empty(shape) for iv in iset_by_control[c]} for c in controls
    }

    for idx in np.ndindex(*shape):
        point_vals = {k: state_grid[k][i] for k, i in zip(grid_axes, idx)}
        states = {k: v for k, v in point_vals.items() if k in arrival}
        obs = {k: v for k, v in point_vals.items() if k in shock_syms}

        # Per-control bounds and seeds, evaluated at this point (each control's
        # pre-state is available once the arrival states / observed shocks are
        # fixed). The optimizer ranges over the stacked control vector, ordered
        # as ``controls``.
        bounds = []
        seed_vec = []
        for c in controls:
            pre = bp.compute_pre_state(c, states, shocks=obs, parameters=params)
            for iv in iset_by_control[c]:
                iset_coord_buf[c][iv][idx] = pre[iv]
            bag = {**params, **obs, **states, **pre}
            lower_func = lower_by_control[c]
            upper_func = upper_by_control[c]
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
            bounds.append((lb, ub))
            # Seed: warm-start > midpoint of finite bounds > x0 fallback (§2).
            if x0_policy is not None:
                seed_vec.append(float(x0_policy[c].values[idx]))
            elif lower_func is not None and upper_func is not None:
                seed_vec.append((lb + ub) / 2)
            else:
                seed_vec.append(x0)

        def negated_value(a):
            ctrl = {c: a[j] for j, c in enumerate(controls)}
            rewards = bp.reward_function(
                states, ctrl, shocks=obs, parameters=params, agent=agent
            )
            r = sum(rewards[s] for s in reward_syms)
            post = bp.post_function(states, ctrl, shocks=obs, parameters=params)
            beta = bp.resolve_discount_factor(post)
            s_next = bp.transition_function(states, ctrl, shocks=obs, parameters=params)
            # Coerce to a plain float: a reward like ``crra_utility`` returns a
            # torch tensor, and scipy.optimize.minimize must see a numpy/python
            # scalar objective (a leaked torch tensor breaks np.asarray on some
            # numpy/torch versions: "'torch.dtype' object has no attribute 'type'").
            return -float(r + beta * continuation_vf(s_next, obs, params))

        res = minimize(negated_value, seed_vec, bounds=bounds)
        if not res.success:
            logging.warning(
                "bellman_step optimization did not converge at %s: %s",
                point_vals,
                res.message,
            )
        for j, c in enumerate(controls):
            policy_buf[c][idx] = res.x[j]
        value_buf[idx] = -res.fun

    coords = {k: state_grid[k] for k in grid_axes}
    value_array = xr.DataArray(value_buf, dims=grid_axes, coords=coords)
    policy_array = {
        c: xr.DataArray(policy_buf[c], dims=grid_axes, coords=coords) for c in controls
    }

    # The decision rule for each control is its policy re-expressed over its own
    # information set (§5); policy_array stays over the state grid so
    # solve_bellman can warm-start the next iterate at the same points.
    dr_from_data = {
        c: ar_from_data(
            _project_to_iset(
                policy_array[c], grid_axes, iset_by_control[c], iset_coord_buf[c], c
            )
        )
        for c in controls
    }

    return dr_from_data, value_array, policy_array


def value_array_to_function(
    value_array: xr.DataArray,
    bp: BellmanPeriod,
    disc_params: Mapping = {},
) -> Callable:
    """
    Rebuild a continuation value function from an iterate's value grid (§4).

    :func:`solve_bellman` feeds iteration *n*'s value grid back as iteration
    *(n+1)*'s continuation. This wraps the grid as a callable in the
    ``bp.compute_value`` convention ``wf(states, shocks, parameters)``, so it
    drops straight into :func:`bellman_step`'s ``continuation_vf`` slot.

    The value grid ranges over the arrival states (its axes). ``wf`` interpolates
    linearly over them and **extrapolates linearly** past the grid edges (via
    :class:`scipy.interpolate.RegularGridInterpolator`), so an off-grid
    next-period state during the backup gets a finite, sloped continuation rather
    than ``NaN`` (which breaks the optimizer) or a flat boundary clamp (which
    zeroes the marginal value of saving and collapses the policy onto its bound).

    .. note::
       Current (deterministic) scope: the value grid is assumed to have no
       observed-shock axes to integrate out — ``disc_params`` and a grid axis
       that is a model shock both raise :class:`NotImplementedError`. Integrating
       observed shocks out of the arrival value (``W(s) = E_obs[V(s, obs)]``)
       lands with the shock-discretization work (design §4, §9 step 6).

    Parameters
    ----------
    value_array : xarray.DataArray
        A gridded value function over arrival states, e.g. the ``value_array``
        returned by :func:`bellman_step`.
    bp : BellmanPeriod
        The recurring period; used to identify shock axes.
    disc_params : Mapping, optional
        Reserved for observed-shock integration (§4); must be empty for now.

    Returns
    -------
    callable
        ``wf(states, shocks, parameters)`` returning the interpolated arrival
        value at ``states``. ``shocks`` and ``parameters`` are accepted for the
        ``bp.compute_value`` calling convention but unused in this scope.
    """
    if disc_params:
        raise NotImplementedError(
            "disc_params (observed-shock integration into the arrival value) is "
            "not yet implemented (design §9 step 6); the continuation is the "
            "deterministic value grid over arrival states."
        )
    axes = list(value_array.dims)
    shock_axes = [ax for ax in axes if ax in set(bp.get_shocks())]
    if shock_axes:
        raise NotImplementedError(
            f"value_array has observed-shock axes {shock_axes} to integrate out; "
            "the arrival-value expectation W(s) = E_obs[V(s, obs)] is not yet "
            "implemented (design §9 step 6)."
        )

    from scipy.interpolate import RegularGridInterpolator

    points = tuple(np.asarray(value_array[ax].values, dtype=float) for ax in axes)
    rgi = RegularGridInterpolator(
        points,
        np.asarray(value_array.values, dtype=float),
        bounds_error=False,
        fill_value=None,  # None -> linear extrapolation past the grid edges
    )

    def wf(states, shocks, parameters):
        cols = [np.asarray(states[ax], dtype=float) for ax in axes]
        scalar = all(c.ndim == 0 for c in cols)
        # Pointwise (not outer-product) query: one row per evaluation point.
        query = np.stack([np.atleast_1d(c).ravel() for c in cols], axis=-1)
        out = rgi(query)
        return float(out[0]) if scalar else out

    return wf


def solve_bellman(
    bp: BellmanPeriod,
    state_grid: Grid,
    *,
    continuation_vf: Callable | None = None,
    agent: str | None = None,
    scope: Mapping = {},
    disc_params: Mapping = {},
    tol: float = 1e-6,
    max_iter: int = 100,
    x0: float = 1.0,
    raise_on_nonconvergence: bool = False,
) -> tuple[dict[str, Callable], xr.DataArray, dict[str, xr.DataArray]]:
    """
    Solve a recurring ``BellmanPeriod`` by value-function iteration (§3).

    Iterates :func:`bellman_step` to a fixed point: each backup uses the previous
    iterate's value grid as its continuation (rebuilt via
    :func:`value_array_to_function`) and warm-starts the per-point optimizer from
    the previous iterate's ``policy_array``. It stops when the sup-norm change in
    the value grid falls below *tol*, or after *max_iter* iterations.

    Iteration 1 uses the terminal (zero) continuation, so
    ``solve_bellman(..., max_iter=1)`` reproduces :func:`bellman_step` under a
    terminal continuation. For an infinite-horizon problem the loop converges
    geometrically (modulus the discount factor) to the stationary solution; for a
    finite horizon of length ``T`` set ``max_iter=T``.

    .. note::
       Current (deterministic) scope, inherited from :func:`bellman_step` and
       :func:`value_array_to_function`: no internal shock discretization
       (``disc_params`` raises :class:`NotImplementedError`); a hidden shock must
       be supplied as a fixed realization via *scope*, and the value grid must
       have no observed-shock axes (design §4, §9 step 6).

    Parameters
    ----------
    bp : BellmanPeriod
        The recurring period providing the model mechanics.
    state_grid : Grid
        A grid over the value function's domain (arrival states and/or observed
        shocks); see :func:`bellman_step`.
    continuation_vf : callable, optional
        Initial continuation guess ``continuation_vf(states, shocks, parameters)``.
        Defaults to the terminal (zero) continuation.
    agent : str, optional
        If given, the period reward sums only this agent's reward symbols.
    scope : Mapping, optional
        Fixed non-shock exogenous values (and, in this scope, any hidden-shock
        realization) merged into the model parameters.
    disc_params : Mapping, optional
        Reserved for internal shock discretization (§4); must be empty for now.
    tol : float, optional
        Convergence tolerance on the sup-norm change in the value grid.
    max_iter : int, optional
        Maximum number of backups.
    x0 : float, optional
        Fallback optimizer seed passed to :func:`bellman_step`.
    raise_on_nonconvergence : bool, optional
        If ``True``, raise :class:`RuntimeError` when the loop hits *max_iter*
        without converging; otherwise emit a :class:`warnings.warn` and return the
        last iterate (the scipy ``OptimizeResult.success`` convention, O5).

    Returns
    -------
    dr_from_data : dict of callable
        One decision rule per control at the fixed point (see :func:`bellman_step`).
    value_array : xarray.DataArray
        The converged value grid. Its ``attrs`` carry ``n_iter``, ``converged``
        (bool), and ``residual`` (the final sup-norm change).
    policy_array : dict of xarray.DataArray
        The gridded optimal control(s) at the fixed point.

    Raises
    ------
    RuntimeError
        If *raise_on_nonconvergence* is ``True`` and the loop does not converge.
    """
    if disc_params:
        raise NotImplementedError(
            "disc_params (internal shock discretization) is not yet implemented "
            "(design §9 step 6); for now supply any hidden-shock realization "
            "as a fixed value via `scope`."
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    cont = continuation_vf if continuation_vf is not None else (lambda s, sh, p: 0.0)
    value_prev = None
    x0_policy = None  # warm-start: previous iterate's optimum
    converged = False
    residual = float("inf")
    for it in range(max_iter):
        dr, value_array, policy_array = bellman_step(
            bp,
            cont,
            state_grid,
            agent=agent,
            scope=scope,
            x0=x0,
            x0_policy=x0_policy,
        )
        if value_prev is not None:
            residual = float(np.abs(value_array - value_prev).max())
            if residual < tol:
                converged = True
                break
        value_prev = value_array
        x0_policy = policy_array  # seed the next backup from this optimum
        cont = value_array_to_function(value_array, bp, disc_params)  # W_n from V_n

    value_array.attrs.update(n_iter=it + 1, converged=converged, residual=residual)
    if not converged:
        msg = (
            f"solve_bellman did not converge in {it + 1} iters "
            f"(residual={residual}); returning last iterate."
        )
        if raise_on_nonconvergence:
            raise RuntimeError(msg)
        warnings.warn(msg)

    return dr, value_array, policy_array
