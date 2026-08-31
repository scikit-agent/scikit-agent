"""
Value function iteration (VFI).

Derive a decision rule, decision value function, and arrival value function for
a single :class:`~skagent.block.DBlock` stage by value function iteration: at each
point of a grid over the decision's information set, solve an exact
:func:`scipy.optimize.minimize` for the control that maximizes the period reward
plus a continuation value.
"""

from skagent.bellman import BellmanPeriod
from skagent.block import DBlock
from skagent.distributions import expected
from skagent.relevance import HIDDEN, MIXED, OBSERVED
from skagent.utils import param_names
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
    ``dr(*[vals[v] for v in iset])``, so a VFI-fitted rule is a drop-in for the
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


#: One coordinate vector per variable, whose cartesian product is the lattice a
#: value array is tabulated over. Unrelated to :class:`skagent.grid.Grid`, which
#: is a batch of scattered points rather than a per-axis specification.
AxisSpec = Mapping[str, Sequence]


def grid_to_data_array(
    grid: AxisSpec = {},  ## TODO: Better data structure here.
):
    """
    Construct a zero-valued ``DataArray`` over the coordinates of a grid.

    Parameters
    ----------
    grid : AxisSpec, optional
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


def solve(block: DBlock, continuation, state_grid: AxisSpec, disc_params={}, scope={}):
    """
    Solve a ``DBlock`` stage by value function iteration.

    At each point of *state_grid*, the optimal control(s) are found with
    :func:`scipy.optimize.minimize`, maximizing the period reward plus the
    *continuation* value of the resulting states. The tabulated optima are then
    interpolated into a decision rule.

    VFI assumes *full observation*: the decision conditions on its complete
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
    state_grid : AxisSpec
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
           denotes fixed, single-valued *parameters* only. Here (legacy VFI
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

    # get_controls() returns a dict[sym, Control]; VFI works with the
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
                lower_bound = feq(*[pre_states[var] for var in param_names(feq)])

            ## get upper bound
            ## assumes only one control currently
            upper_bound = 1e12  # a very high number
            feq = block.dynamics[controls[0]].upper_bound

            if feq is not None:
                upper_bound = feq(*[pre_states[var] for var in param_names(feq)])

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
    arr_vf = block.get_arrival_value_function(
        disc_params, dr_from_data, continuation, calibration=scope
    )

    return dr_from_data, dec_vf, arr_vf


# Sentinels for an "open" (effectively infinite) bound, symmetric with legacy
# ``solve``. A bound this large is treated as absent when seeding ``x0``.
_LOWER_OPEN = -1e12
_UPPER_OPEN = 1e12

# A coordinate is treated as varying along a grid axis when its spread there
# exceeds this; below it, the axis is considered flat (invariant). Flat absolute
# at the test ATOL scale; relative/absolute hardening is deferred to Phase 3.
_PROJ_TOL = 1e-3

# Two multi-start optima count as equally good -- so the argmax is *not
# identified* if they disagree by more than _PROJ_TOL -- when their objectives
# differ by less than this, relative to the objective's own scale. A constant
# objective, the case this exists for, returns each seed at exactly equal value.
_TIE_TOL = 1e-10


def _gather_and_fit(policy_da, grid_axes, iset_var, coord, identified):
    """Fit a policy as a 1-D function of an information-set variable.

    Collects every ``(coordinate, control)`` pair the backup produced, sorts by
    the coordinate, and returns a 1-D :class:`~xarray.DataArray` ready for
    :func:`ar_from_data`. Used where the coordinate varies along several grid
    axes, so no axis can be relabelled to it. The result carries a sorted
    irregular coordinate rather than a regular grid.

    *identified* masks out points whose optimum is not pinned down; ``None``
    treats every point as identified.

    Raises
    ------
    ValueError
        If two points sharing a coordinate disagree in the control by more than
        ``_PROJ_TOL``, so the control depends on more than *iset_var*; or if no
        point is identified.
    """
    coords = np.asarray(coord, dtype=float).ravel()
    policy = np.asarray(policy_da.transpose(*grid_axes).values, dtype=float).ravel()

    if identified is not None:
        keep = np.asarray(identified, dtype=bool).ravel()
        coords, policy = coords[keep], policy[keep]
    if coords.size == 0:
        raise ValueError(
            "No grid point has an identified optimum, so the policy over "
            f"'{iset_var}' cannot be built."
        )

    order = np.argsort(coords, kind="stable")
    coords, policy = coords[order], policy[order]

    # Equal coordinate => equal control. The coordinate tolerance is tight, since
    # this asks whether two samples are the *same* point of the information set,
    # while the control tolerance is the projection tolerance.
    same = np.diff(coords) <= 1e-9 * (1.0 + np.abs(coords[:-1]))
    disagrees = same & (np.abs(np.diff(policy)) > _PROJ_TOL)
    if disagrees.any():
        i = int(np.argmax(disagrees))
        raise ValueError(
            f"Two grid points share information-set coordinate "
            f"{iset_var}={coords[i]:.6g} but disagree in the control "
            f"({policy[i]:.6g} vs {policy[i + 1]:.6g}), so it depends on more "
            f"than '{iset_var}'."
        )

    # Interpolation needs a strictly increasing coordinate. Coincident samples
    # agree to within tolerance by the check above, so keeping one is enough.
    unique = np.ones(coords.size, dtype=bool)
    unique[1:] = ~same
    return xr.DataArray(
        policy[unique], dims=[iset_var], coords={iset_var: coords[unique]}
    )


def _project_to_iset(
    policy_da: xr.DataArray,
    grid_axes: Sequence[str],
    iset: Sequence[str],
    iset_coords: Mapping[str, np.ndarray],
    control_sym: str,
    identified: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Re-express a grid-tabulated policy as a function of a control's information
    set, ready for :func:`ar_from_data`.

    The backup grids over the value function's domain (arrival states + observed
    shocks); a control conditions on its own ``iset``. Each iset variable must
    track exactly one grid axis: either it *is* that axis, or it is a derived
    pre-state (e.g. ``m = a·R + y``) that varies, strictly monotonically, only
    along it — in which case that axis is relabeled to the variable and
    recoordinated onto its computed values. Any leftover grid axis is outside the
    iset, so the optimum must be invariant along it; it is dropped.

    Where the objective is constant in the controls the optimizer just returns its
    seed, so that point says nothing about dependence on a dropped axis — generic
    at an absorbing state with zero reward and continuation (D-3's dead
    ``liv = 0`` slice). *identified* marks the points whose optimum is pinned down;
    the invariance check and the surviving slice are taken over those only.
    ``None`` treats every point as identified.

    Raises ``NotImplementedError`` if an iset variable varies along more than one
    grid axis, or two variables share an axis (general scattered reindexing, out
    of scope); ``ValueError`` if a derived map is non-monotone, a dropped axis is
    non-invariant, or the policy is nowhere identified along a dropped axis (the
    reprojection would be ill-posed).
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
            if len(varying) > 1:
                # The coordinate tracks no single axis, so there is nothing to
                # relabel; gather its samples instead. Only well posed as a
                # 1-D fit, hence the single-variable restriction.
                if len(iset) == 1:
                    return _gather_and_fit(policy_da, grid_axes, iv, coord, identified)
                raise NotImplementedError(
                    f"Information-set variable '{iv}' of control '{control_sym}' "
                    f"varies along grid axes {[grid_axes[k] for k in varying]}, "
                    f"and its information set {list(iset)} has other variables. "
                    "Gathering a multi-axis coordinate is implemented only for a "
                    "single-variable information set, since each slice of the "
                    "remaining variables would gather a different coordinate."
                )
            if len(varying) != 1:
                raise NotImplementedError(
                    f"Information-set variable '{iv}' of control '{control_sym}' "
                    f"does not vary along any grid axis, so the grid pins it to a "
                    "single value and cannot support a rule over it."
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

    mask = None
    if identified is not None:
        mask = xr.DataArray(
            np.asarray(identified, dtype=bool), dims=grid_axes, coords=policy_da.coords
        )

    for ax in grid_axes:
        if ax in claimed:
            continue
        # Unidentified points testify to nothing about dependence on ``ax``, so
        # they are masked out of both the check and the surviving slice.
        witness = da if mask is None else da.where(mask)
        if mask is not None and not bool(mask.any(dim=ax).all()):
            raise ValueError(
                f"Control '{control_sym}'s optimum is unidentified at every point "
                f"of grid axis '{ax}', so no value along it can be carried forward."
            )
        spread = float(np.abs(witness.max(dim=ax) - witness.min(dim=ax)).max())
        if spread > _PROJ_TOL:
            raise ValueError(
                f"Control '{control_sym}'s optimum varies along grid axis "
                f"'{ax}' (spread {spread:.3g}), which is outside its "
                f"information set {list(iset)}."
            )
        # Surviving values agree to within _PROJ_TOL, so any one represents the
        # line; ``isel(0)`` when nothing is masked, i.e. exactly the plain drop.
        if mask is None:
            da = da.isel({ax: 0})
        else:
            da, mask = witness.max(dim=ax), mask.any(dim=ax)

    da = da.rename({ax: iv for ax, iv in claimed.items() if ax != iv})
    return da.assign_coords(derived_coords).transpose(*iset)


def _shock_nodes(bp, shock, disc_params):
    """The discretization nodes of one *shock*, as a 1-D array.

    Built from ``bp.calibration`` rather than from a caller's ``scope`` so the
    nodes match the weights :func:`_integrate_observed_shocks` recomputes when it
    integrates the axis back out; a mismatch there raises rather than
    mis-weighting.
    """
    from skagent.block import construct_shocks

    dist = construct_shocks({shock: bp.get_shocks()[shock]}, dict(bp.calibration))[
        shock
    ]
    disc = (
        dist.discretize(**disc_params[shock])
        if shock in disc_params
        else dist.discretize()
    )
    return np.asarray(disc.points, dtype=float).ravel()


def _resolve_shock_roles(bp, params, grid_axes):
    """Resolve each shock's information role once for the whole backup.

    A role is a property of the block, not of a grid point, so it is settled
    before the per-point loop and the loop consumes only the resulting partition.

    Returns ``{shock: role}`` over the shocks the criterion governs -- those not
    pinned to a fixed value in *params*, which are caller-supplied realizations
    rather than variables to integrate.

    Raises
    ------
    NotImplementedError
        If a shock is ``MIXED``, so posing the declared problem needs the
        conditional law of the shock given the information set; or if its role
        differs across controls, which this backup cannot represent because it
        grids or integrates a shock for the period as a whole.
    ValueError
        If a shock already on the state grid is not ``OBSERVED``. A grid axis
        conditions the decision on that shock's realization, so gridding one no
        information set accounts for would solve an information structure the
        block does not declare.

    Notes
    -----
    The criterion reads the block's structure from ``bp.calibration`` alone, not
    from *params*. Pinning a value in a caller's ``scope`` fixes it for this solve
    but does not make the symbol a structural constant: an arrival state pinned
    there is still reassigned in-period by dynamics that may depend on a shock, so
    treating it as a parameter would sever that route and misclassify the shock.
    *params* governs only which shocks are fixed realizations.
    """
    per_control = bp.block.shock_roles(bp.calibration)
    resolved = {}

    for shock in bp.get_shocks():
        if shock in params:
            continue
        by_control = {c: roles[shock] for c, roles in per_control.items()}
        distinct = set(by_control.values())

        if MIXED in distinct:
            raise NotImplementedError(
                f"Shock '{shock}' is partly revealed by an information set that "
                "conditions on it while also reaching the objective around that "
                "information set, so posing the declared problem requires the "
                "conditional law of the shock given the information set. Either "
                "add it to the information set of every control that sees it, or "
                "keep it out of the reward and transition."
            )
        if len(distinct) > 1:
            raise NotImplementedError(
                f"Shock '{shock}' has different information roles across controls "
                f"({by_control}); this backup grids or integrates a shock for the "
                "period as a whole, so it cannot serve controls that see it "
                "differently."
            )
        resolved[shock] = distinct.pop()

    for shock, role in resolved.items():
        if shock in grid_axes and role != OBSERVED:
            raise ValueError(
                f"Shock '{shock}' is a state-grid axis, which conditions the "
                f"decision on its realization, but the block makes it '{role}': "
                "no control's information set accounts for it. Drop it from the "
                "grid so it is integrated out, or widen an information set."
            )

    return resolved


def _discretize_shocks(bp, shock_syms, params, disc_params):
    """Build the joint discrete distribution over *shock_syms*.

    The shocks are pulled from the block (constructed from *params* if still in
    constructor-tuple form), discretized — with the per-shock ``disc_params``
    arguments where given, else the distribution's default — and combined into a
    single joint :class:`DiscreteDistribution`. Already-discrete shocks (e.g.
    ``Bernoulli``) discretize to themselves, so ``disc_params`` is optional per
    shock.

    Returns ``(joint_dist, means)`` where *means* maps each shock symbol to the
    mean of its discretized marginal — used to fix the shock when computing a
    control's pre-state / bounds, which must be a single value even though the
    objective integrates over the whole distribution.
    """
    from skagent.block import construct_shocks
    from skagent.distributions import (
        DiscreteDistributionLabeled,
        combine_indep_dstns,
    )

    shock_data = {s: bp.get_shocks()[s] for s in shock_syms}
    constructed = construct_shocks(shock_data, params)

    labeled = []
    means = {}
    for s in shock_syms:
        dist = constructed[s]
        disc = (
            dist.discretize(**disc_params[s]) if s in disc_params else dist.discretize()
        )
        labeled.append(DiscreteDistributionLabeled.from_unlabeled(disc, var_names=[s]))
        pts = np.asarray(disc.points, dtype=float)
        wts = np.asarray(disc.weights, dtype=float)
        means[s] = float(np.sum(pts * wts) / np.sum(wts))
    return combine_indep_dstns(*labeled), means


# Two evaluations pin an affine map; a third checks the affinity assumption.
_ABC_PROBES = (1.0, 2.0, 3.0)


def _tighten_bounds_to_grid(bp, control, states, obs, params, grid_box, lb, ub):
    """Tighten ``(lb, ub)`` so the successor arrival state stays in *grid_box*.

    Implements the optional artificial borrowing constraint:
    confining each next-period arrival state to the value grid guarantees the
    continuation is only ever *interpolated*, never linearly *extrapolated* past
    a grid edge into a region it cannot represent. The grid's lower edge thereby
    acts as a slack artificial state (borrowing) constraint.

    For a single control whose successor ``a'(c)`` is **affine** in ``c`` (e.g.
    ``a' = m − c``), the state-box constraint ``grid_min ≤ a' ≤ grid_max``
    inverts *exactly* to a control interval — so this stays a plain box-bound
    tweak (L-BFGS-B unchanged), no optimizer constraints needed. The affine map
    is recovered from two probe evaluations and verified by a third; a nonlinear
    successor raises: treating ``a'(c)`` as a black box and root-solving
    ``a'(c) = grid_edge`` is not implemented.
    """
    probe = {
        p: bp.transition_function(states, {control: p}, shocks=obs, parameters=params)
        for p in _ABC_PROBES
    }
    for k, (lo, hi) in grid_box.items():
        a0, a1, a2 = (float(np.asarray(probe[p][k])) for p in _ABC_PROBES)
        slope = a1 - a0  # (a'(2) − a'(1)) / (2 − 1)
        intercept = a0 - slope * _ABC_PROBES[0]
        # Affinity guard: the third probe must lie on the line built from the
        # first two. A nonlinear successor is outside this fast path.
        if not np.isfinite(a2) or abs(
            a2 - (intercept + slope * _ABC_PROBES[2])
        ) > 1e-8 * (1.0 + abs(a2)):
            raise NotImplementedError(
                f"artificial_borrowing_constraint: successor arrival state '{k}' "
                f"is not affine in control '{control}', so the state-grid box does "
                "not invert to a control interval; root-solving a general monotone "
                "successor is not implemented."
            )
        if abs(slope) < 1e-12:
            continue  # this successor axis does not depend on the control
        c_at_lo = (lo - intercept) / slope
        c_at_hi = (hi - intercept) / slope
        c_min, c_max = sorted((c_at_lo, c_at_hi))
        lb, ub = max(lb, c_min), min(ub, c_max)
    return lb, ub


def bellman_step(
    bp: BellmanPeriod,
    continuation_vf: Callable,
    state_grid: AxisSpec,
    *,
    agent: str | None = None,
    scope: Mapping = {},
    disc_params: Mapping = {},
    x0: float = 1.0,
    x0_policy: Mapping[str, xr.DataArray] | None = None,
    artificial_borrowing_constraint: bool = False,
) -> tuple[dict[str, Callable], xr.DataArray, dict[str, xr.DataArray]]:
    """
    One exact value backup over *state_grid* on the ``BellmanPeriod`` protocol.

    This is the per-iteration update of value-function iteration: at each grid
    point the optimal control is found with :func:`scipy.optimize.minimize`,
    maximizing the period reward plus the discounted *continuation* value of the
    resulting arrival states. Under a terminal (zero) continuation,
    ``continuation_vf = lambda s, sh, p: 0.0``, the result is the single-step
    solution; the value-iteration wrapper ``solve_bellman`` iterates it to a
    fixed point.

    Unlike legacy :func:`solve` (which rides the ``DBlock`` continuation API and
    folds the discount factor into the continuation), this speaks the
    ``BellmanPeriod`` protocol the rest of the torch stack uses, with an explicit
    discount factor and multi-reward summation, and is empty-shock-safe.

    Shocks are handled by their information role, which is derived from the
    block's own structure (:mod:`skagent.relevance`) rather than inferred from
    the grid the caller supplies. A shock some control's information set accounts
    for becomes a grid axis over its discretization nodes, so its pre-state and
    bounds are computed per realization; the rest are integrated out inside the
    per-point ``max`` via internal discretization (:func:`_discretize_shocks` +
    ``expected``), so an optimum characterized by an expectation over an
    unobserved shock is in scope. Any shock may instead be pinned to a fixed
    realization by supplying it in *scope*, which takes precedence over both.

    .. note::
       Current scope: one or more controls, jointly optimized by
       :func:`scipy.optimize.minimize` over the stacked control vector with
       per-control bounds — restarted from each seed candidate, keeping the best
       optimum — each policy then reprojected onto its own information set
       (:func:`_project_to_iset`). A control's pre-state and bounds are evaluated
       with each integrated-out shock fixed at its (discretized) mean, since a
       single value is required there even though the objective integrates the
       shock. That is exact when the pre-state does not depend on such a shock,
       and a shock the pre-state *does* depend on is a grid axis rather than an
       integrated one, so it does not arise for a well-formed block.

    Parameters
    ----------
    bp : BellmanPeriod
        The recurring period providing the model mechanics.
    continuation_vf : callable
        The continuation value function, called ``continuation_vf(states, shocks,
        parameters)`` on the next-period arrival states (the ``bp.compute_value``
        convention). Terminal continuation is ``lambda s, sh, p: 0.0``.
    state_grid : AxisSpec
        The shared backup grid over arrival states: one axis per variable. Axes
        for the shocks an information set accounts for are added automatically
        from their discretization nodes, so supplying one is optional and
        idempotent. This grid covers the variables the Bellman loop iterates over
        and is not necessarily equal to any individual control's information set
        (a control's iset may be a strict subset). For an empty grid, pass ``{}``
        — the empty mapping, not :class:`skagent.grid.Grid`, whose
        ``from_config({})`` raises.
    agent : str, optional
        If given, the period reward sums only this agent's reward symbols.
    scope : Mapping, optional
        Fixed non-shock exogenous values merged into the model parameters. A
        shock supplied here is pinned to that fixed realization instead of being
        gridded or integrated. Note this fixes a value for the solve; it does not
        make the symbol a model parameter, so it does not affect any shock's
        information role.
    disc_params : Mapping, optional
        Per-shock discretization arguments, keyed by shock symbol (e.g.
        ``{"theta": {"N": 7}}``), forwarded to that shock's
        ``Distribution.discretize`` whether it becomes a grid axis or is
        integrated out. A shock without an entry uses its distribution's default
        discretization (exact for already-discrete shocks).
    x0 : float, optional
        A modest optimizer seed, clamped into each control's bounds. One of the
        multi-start candidates at every point, not just a fallback: it is what
        covers a box whose midpoint is far from the optimum, e.g. the natural
        borrowing limit's ``[0, m + H]``.
    x0_policy : Mapping[str, DataArray], optional
        Warm-start seeds keyed by control symbol (e.g. a previous iterate's
        ``policy_array``); supplies the first multi-start candidate at each grid
        point, and wins ties. Supplied by :func:`solve_bellman`.
    artificial_borrowing_constraint : bool, optional
        When ``True``, tighten each control's bounds so the next-period arrival
        state stays inside the state grid (:func:`_tighten_bounds_to_grid`), an
        artificial state (borrowing) limit at the grid's lower edge.
        This keeps the continuation interpolated rather than extrapolated past
        the grid edges, so value iteration cannot ride a control bound by
        over-crediting off-grid successors. Single control with an affine
        successor only (raises otherwise). The limit must be *slack* at the
        states of interest (it is just the grid floor), or it biases the policy
        where it binds.

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

    grid_axes = list(state_grid.keys())
    shock_syms = set(bp.get_shocks())
    arrival = bp.arrival_states

    # Per-control static info: controls may have *different* information sets and
    # bounds, so everything per-control is keyed by control symbol. The state
    # grid is the value function's domain (arrival states + observed shocks),
    # shared across controls; each control's policy is projected onto its own
    # iset afterwards.
    iset_by_control = {c: bp.block.dynamics[c].iset for c in controls}
    lower_by_control = {c: bp.block.dynamics[c].lower_bound for c in controls}
    upper_by_control = {c: bp.block.dynamics[c].upper_bound for c in controls}

    params = {**bp.calibration, **scope}

    # Each shock's information role, resolved once from the block's own structure
    # rather than inferred from what the caller happened to grid. A shock pinned
    # in ``scope`` is a fixed realization and is governed by neither branch below.
    roles = _resolve_shock_roles(bp, params, grid_axes)

    # A shock some information set accounts for becomes a node axis, so the
    # pre-state and each control's bounds are computed *per node* rather than at
    # one summary value: the control conditions on its realization, so the
    # decision problem is posed separately for each. Idempotent for a shock the
    # caller already gridded. ``_integrate_observed_shocks`` integrates these axes
    # back out into the arrival value.
    state_grid = dict(state_grid)
    for shock, role in roles.items():
        if role == OBSERVED and shock not in grid_axes:
            state_grid[shock] = _shock_nodes(bp, shock, disc_params)
            grid_axes.append(shock)

    # What remains is integrated out *inside* the per-point max, since no
    # information set says anything about it. Such a shock is fixed at its mean
    # when computing a pre-state or bound, where a single value is required, and
    # swept over its nodes only inside the objective.
    hidden_syms = [s for s, role in roles.items() if role == HIDDEN]
    disc_hidden = None
    hidden_mean = {}
    if hidden_syms:
        disc_hidden, hidden_mean = _discretize_shocks(
            bp, hidden_syms, params, disc_params
        )
    hidden_names = list(disc_hidden.var_names) if disc_hidden is not None else []

    reward_syms = bp.get_reward_syms(agent)

    # Artificial borrowing constraint: the box of arrival-state grid axes
    # that successors must stay within. Single-control only for now — with >1
    # control the successor couples them into a joint constraint that no longer
    # reduces to per-control box bounds (that would need optimizer-level
    # inequality constraints, e.g. SLSQP, rather than a bound tweak).
    grid_box = None
    if artificial_borrowing_constraint:
        if len(controls) != 1:
            raise NotImplementedError(
                "artificial_borrowing_constraint supports a single control for "
                "now; a multi-control successor is a coupled (joint) constraint, "
                "not per-control box bounds; that is not supported."
            )
        grid_box = {
            k: (float(np.min(state_grid[k])), float(np.max(state_grid[k])))
            for k in grid_axes
            if k in arrival
        }

    shape = tuple(len(state_grid[k]) for k in grid_axes)
    value_buf = np.empty(shape)
    policy_buf = {c: np.empty(shape) for c in controls}
    # Tabulate each control's information-set coordinates over the grid, in case
    # a variable is a derived pre-state that must be reprojected.
    iset_coord_buf = {
        c: {iv: np.empty(shape) for iv in iset_by_control[c]} for c in controls
    }
    # Whether each point's optimum is identified -- only worth probing for
    # when some control's iset is narrower than the grid, so an axis may be
    # dropped; otherwise no optimum has to testify about one.
    check_identified = any(len(iset_by_control[c]) < len(grid_axes) for c in controls)
    identified_buf = np.ones(shape, dtype=bool)

    for idx in np.ndindex(*shape):
        point_vals = {k: state_grid[k][i] for k, i in zip(grid_axes, idx)}
        states = {k: v for k, v in point_vals.items() if k in arrival}
        obs = {k: v for k, v in point_vals.items() if k in shock_syms}

        # Shocks seen when computing a control's pre-state and bounds: observed
        # nodes at this point, plus each hidden shock fixed at its mean (a single
        # value is required here; the objective integrates the hidden shocks).
        pre_shocks = {**obs, **hidden_mean}

        # Per-control bounds and seeds, evaluated at this point (each control's
        # pre-state is available once the arrival states / observed shocks are
        # fixed). The optimizer ranges over the stacked control vector, ordered
        # as ``controls``.
        bounds = []
        seed_cands = []
        for c in controls:
            pre = bp.compute_pre_state(c, states, shocks=pre_shocks, parameters=params)
            for iv in iset_by_control[c]:
                iset_coord_buf[c][iv][idx] = pre[iv]
            bag = {**params, **pre_shocks, **states, **pre}
            lower_func = lower_by_control[c]
            upper_func = upper_by_control[c]
            lb = (
                lower_func(*[bag[v] for v in param_names(lower_func)])
                if lower_func is not None
                else _LOWER_OPEN
            )
            ub = (
                upper_func(*[bag[v] for v in param_names(upper_func)])
                if upper_func is not None
                else _UPPER_OPEN
            )
            if grid_box is not None:
                lb, ub = _tighten_bounds_to_grid(
                    bp, c, states, pre_shocks, params, grid_box, lb, ub
                )
            bounds.append((lb, ub))
            # Multi-start candidates, in tie-breaking order: the warm start,
            # the midpoint of finite bounds, and the modest ``x0`` clamped into the
            # box. Every one is tried; a bad seed can no longer decide the answer.
            cands = [] if x0_policy is None else [float(x0_policy[c].values[idx])]
            if lower_func is not None and upper_func is not None:
                cands.append((lb + ub) / 2)
            cands.append(min(max(x0, lb), ub))
            seed_cands.append(cands)

        def value_at(a, extra_shocks):
            # One evaluation of the backup objective at control vector ``a`` and
            # a single hidden-shock realization (``extra_shocks``; empty when
            # there are no hidden shocks to integrate).
            ctrl = {c: a[j] for j, c in enumerate(controls)}
            sh = {**obs, **extra_shocks}
            # One block pass: the ex post values carry the reward symbols, the
            # discount factor and the next-period arrival states alike.
            post = bp.post_function(states, ctrl, shocks=sh, parameters=params)
            r = sum(post[s] for s in reward_syms)
            beta = bp.resolve_discount_factor(post)
            s_next = bp.select_arrival_states(post)
            return float(r + beta * continuation_vf(s_next, sh, params))

        def negated_value(a):
            if disc_hidden is None:
                return -value_at(a, {})

            # Integrate the hidden shocks inside the max: E_hidden[objective].
            # The realization arrives keyed by shock name (indexed by name so the
            # single- and multi-shock joint distributions share one code path).
            def obj(shock_value_array):
                extra = {s: shock_value_array[s] for s in hidden_names}
                return value_at(a, extra)

            return -float(expected(obj, disc_hidden))

        # Start vectors are *aligned*, not crossed: candidate j for every control
        # at once, so the count is the longest candidate list rather than its power
        # (a control with fewer candidates repeats its last). Duplicates are
        # dropped, since restarting from the same point learns nothing.
        starts = []
        for j in range(max(len(s) for s in seed_cands)):
            v = np.array([s[min(j, len(s) - 1)] for s in seed_cands])
            if not any(np.allclose(v, w, atol=_PROJ_TOL) for w in starts):
                starts.append(v)

        # Optimize from every start candidate and keep the best result. The per-point
        # objective need not be unimodal for general blocks, so skipping a candidate
        # based on its seed value can miss a better optimum.
        results = [minimize(negated_value, v, bounds=bounds) for v in starts]
        res = min(results, key=lambda r: r.fun)  # ties keep the earliest candidate
        if not res.success:
            logging.warning(
                "bellman_step optimization did not converge at %s: %s",
                point_vals,
                res.message,
            )
        for j, c in enumerate(controls):
            policy_buf[c][idx] = res.x[j]
        value_buf[idx] = -res.fun

        # Identifiability, free from the restarts: an equally good optimum at
        # a materially different control vector means the argmax is not pinned
        # down -- what happens where the objective is constant, since each start
        # then returns its own seed. A single start cannot show this, so it leaves
        # the point (optimistically) identified.
        if check_identified:
            identified_buf[idx] = not any(
                abs(r.fun - res.fun) <= _TIE_TOL * (1.0 + abs(res.fun))
                and float(np.max(np.abs(r.x - res.x))) > _PROJ_TOL
                for r in results
            )

    coords = {k: state_grid[k] for k in grid_axes}
    value_array = xr.DataArray(value_buf, dims=grid_axes, coords=coords)
    policy_array = {
        c: xr.DataArray(policy_buf[c], dims=grid_axes, coords=coords) for c in controls
    }

    # The decision rule for each control is its policy re-expressed over its own
    # information set; policy_array stays over the state grid so
    # solve_bellman can warm-start the next iterate at the same points.
    dr_from_data = {
        c: ar_from_data(
            _project_to_iset(
                policy_array[c],
                grid_axes,
                iset_by_control[c],
                iset_coord_buf[c],
                c,
                identified=identified_buf if check_identified else None,
            )
        )
        for c in controls
    }

    return dr_from_data, value_array, policy_array


def _integrate_observed_shocks(value_array, bp, shock_axes, disc_params):
    """Integrate observed-shock axes out of a decision value grid.

    Returns ``W(s) = E_obs[V(s, obs)]`` over the arrival-state axes, summing each
    shock axis against the weights of that shock's discretized distribution. The
    axis coordinate must equal the discretization node values (in order), since
    the weights are matched to the nodes positionally; a grid axis built from
    other values raises :class:`ValueError` rather than mis-weighting.
    """
    from skagent.block import construct_shocks

    constructed = construct_shocks(
        {s: bp.get_shocks()[s] for s in shock_axes}, dict(bp.calibration)
    )
    for ax in shock_axes:
        dist = constructed[ax]
        disc = (
            dist.discretize(**disc_params[ax])
            if ax in disc_params
            else dist.discretize()
        )
        nodes = np.asarray(disc.points, dtype=float)
        weights = np.asarray(disc.weights, dtype=float)
        axis_coords = np.asarray(value_array[ax].values, dtype=float)
        if axis_coords.shape != nodes.shape or not np.allclose(axis_coords, nodes):
            raise ValueError(
                f"Observed-shock axis '{ax}' has coordinate {axis_coords} but its "
                f"discretization nodes are {nodes}; the grid axis must be built "
                "from the shock's discretization nodes so the expectation weights "
                "align. Adjust the grid axis or the disc_params."
            )
        w_da = xr.DataArray(weights, dims=[ax], coords={ax: value_array[ax]})
        value_array = (value_array * w_da).sum(ax)
    return value_array


def value_array_to_function(
    value_array: xr.DataArray,
    bp: BellmanPeriod,
    disc_params: Mapping = {},
) -> Callable:
    """
    Rebuild a continuation value function from an iterate's value grid.

    :func:`solve_bellman` feeds iteration *n*'s value grid back as iteration
    *(n+1)*'s continuation. This wraps the grid as a callable in the
    ``bp.compute_value`` convention ``wf(states, shocks, parameters)``, so it
    drops straight into :func:`bellman_step`'s ``continuation_vf`` slot.

    The decision value grid ranges over arrival states and any *observed*-shock
    axes. Those shock axes are first integrated out into the arrival value
    ``W(s) = E_obs[V(s, obs)]``, using the weights of the same discretized
    distribution that produced the axis nodes; a grid built from other node
    values raises :class:`ValueError`. ``wf`` then interpolates linearly over the
    remaining arrival-state axes and **extrapolates linearly** past the grid
    edges (via :class:`scipy.interpolate.RegularGridInterpolator`), so an
    off-grid next-period state during the backup gets a finite, sloped
    continuation rather than ``NaN`` (which breaks the optimizer) or a flat
    boundary clamp (which zeroes the marginal value of saving and collapses the
    policy onto its bound).

    When the value grid has no observed-shock axes (the deterministic and
    hidden-shock-only cases, where the backup already integrated any hidden
    shocks), the expectation step is a no-op and the value grid over arrival
    states *is* ``W``.

    ``wf`` reads only the axes it has, so it raises :class:`ValueError` rather than
    silently discard an arrival state that the value grid has no axis for — with
    D-3's survival state ``liv`` left off the grid, the mortality discount cancels
    out of the backup and the loop converges to the no-mortality policy, a
    plausible value for a different model.

    Parameters
    ----------
    value_array : xarray.DataArray
        A gridded decision value function, e.g. the ``value_array`` returned by
        :func:`bellman_step`; its axes are arrival states and any observed-shock
        nodes.
    bp : BellmanPeriod
        The recurring period; used to identify and discretize the shock axes.
    disc_params : Mapping, optional
        Per-shock discretization arguments for the observed-shock axes, keyed by
        shock symbol. A shock axis without an entry uses its distribution's
        default discretization.

    Returns
    -------
    callable
        ``wf(states, shocks, parameters)`` returning the interpolated arrival
        value at ``states``. ``shocks`` and ``parameters`` are accepted for the
        ``bp.compute_value`` calling convention but unused.
    """
    axes = list(value_array.dims)
    shock_axes = [ax for ax in axes if ax in set(bp.get_shocks())]
    if shock_axes:
        value_array = _integrate_observed_shocks(
            value_array, bp, shock_axes, disc_params
        )
        axes = list(value_array.dims)

    from scipy.interpolate import RegularGridInterpolator

    points = tuple(np.asarray(value_array[ax].values, dtype=float) for ax in axes)
    rgi = RegularGridInterpolator(
        points,
        np.asarray(value_array.values, dtype=float),
        bounds_error=False,
        fill_value=None,  # None -> linear extrapolation past the grid edges
    )

    # Arrival states this grid cannot represent; being handed one is an error.
    ungridded = sorted(set(bp.arrival_states) - set(axes))

    def wf(states, shocks, parameters):
        dropped = [k for k in ungridded if k in states]
        if dropped:
            raise ValueError(
                f"Continuation was rebuilt from a value grid over {axes}, which has "
                f"no axis for arrival state(s) {dropped}, so it cannot represent "
                "any dependence on them. Include them in the state grid."
            )
        cols = [np.asarray(states[ax], dtype=float) for ax in axes]
        scalar = all(c.ndim == 0 for c in cols)
        # Pointwise (not outer-product) query: one row per evaluation point.
        query = np.stack([np.atleast_1d(c).ravel() for c in cols], axis=-1)
        out = rgi(query)
        return float(out[0]) if scalar else out

    return wf


def solve_bellman(
    bp: BellmanPeriod,
    state_grid: AxisSpec,
    *,
    continuation_vf: Callable | None = None,
    agent: str | None = None,
    scope: Mapping = {},
    disc_params: Mapping = {},
    tol: float = 1e-6,
    max_iter: int = 100,
    x0: float = 1.0,
    raise_on_nonconvergence: bool = False,
    artificial_borrowing_constraint: bool = False,
) -> tuple[dict[str, Callable], xr.DataArray, dict[str, xr.DataArray]]:
    """
    Solve a recurring ``BellmanPeriod`` by value-function iteration.

    Iterates :func:`bellman_step` to a fixed point: each backup uses the previous
    iterate's value grid as its continuation (rebuilt via
    :func:`value_array_to_function`) and offers the previous iterate's
    ``policy_array`` as a per-point warm start — one multi-start candidate among
    the others, so a collapsed iterate cannot entrench itself by seeding the
    next backup at its own bound. It stops when the sup-norm change in the value
    grid falls below *tol*, or after *max_iter* iterations.

    Iteration 1 uses the terminal (zero) continuation, so
    ``solve_bellman(..., max_iter=1)`` reproduces :func:`bellman_step` under a
    terminal continuation. For an infinite-horizon problem the loop converges
    geometrically (modulus the discount factor) to the stationary solution; for a
    finite horizon of length ``T`` set ``max_iter=T``.

    The stopping rule assumes an infinite horizon. A finite-horizon problem is
    structurally indistinguishable from an infinite-horizon one, so a run that
    reaches *max_iter* is reported as non-convergence in both cases: at a
    caller-specified horizon that report is expected rather than a failure, and
    the residual is a sup-norm change between iterates, not an error against the
    ``T``-period answer.

    A period with no arrival states is not a dynamic problem — nothing carries
    between periods — and is refused. Such a block is solved statically, by
    :class:`skagent.algos.best_response.TabularBestResponseSolver` or
    :func:`skagent.solver.solve_multiple_controls`. Every arrival state must be
    an axis of *state_grid*, since the continuation is rebuilt from the value
    grid and cannot represent dependence on a variable the grid has no axis for.

    Shocks are discretized internally: *disc_params* is threaded into every
    backup (hidden shocks integrated inside the max) and into
    :func:`value_array_to_function` (observed-shock axes integrated into the
    arrival value between iterations).

    Parameters
    ----------
    bp : BellmanPeriod
        The recurring period providing the model mechanics.
    state_grid : AxisSpec
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
        Per-shock discretization arguments, threaded into each backup (for
        hidden shocks) and into :func:`value_array_to_function` (for observed
        shocks); see :func:`bellman_step`.
    tol : float, optional
        Convergence tolerance on the sup-norm change in the value grid.
    max_iter : int, optional
        Maximum number of backups.
    x0 : float, optional
        Modest multi-start seed candidate passed to :func:`bellman_step`.
    raise_on_nonconvergence : bool, optional
        If ``True``, raise :class:`RuntimeError` when the loop hits *max_iter*
        with the sup-norm change still above *tol*; otherwise emit a
        :class:`warnings.warn` and return the last iterate (the scipy
        ``OptimizeResult.success`` convention, O5). At a finite horizon set
        through *max_iter* this condition is expected, so leave it ``False``.
    artificial_borrowing_constraint : bool, optional
        Forwarded to :func:`bellman_step`: confine next-period arrival states to
        the state grid (grid edge = slack artificial borrowing limit),
        so the rebuilt continuation is never extrapolated off-grid.

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
    ValueError
        If *max_iter* is less than 1, or if the period has no arrival states.
    RuntimeError
        If *raise_on_nonconvergence* is ``True`` and the loop does not converge.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    if not bp.arrival_states:
        raise ValueError(
            "solve_bellman iterates a dynamic program to a fixed point, and a "
            "period with no arrival states is not dynamic: nothing carries "
            "between periods, so there is no fixed point to seek. Solve the "
            "block statically instead — TabularBestResponseSolver "
            "(skagent.algos.best_response) for a tabular policy over finitely "
            "many observations, or solve_multiple_controls (skagent.solver) for "
            "a differentiable one."
        )

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
            disc_params=disc_params,
            x0=x0,
            x0_policy=x0_policy,
            artificial_borrowing_constraint=artificial_borrowing_constraint,
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
            f"solve_bellman reached max_iter={it + 1} with a sup-norm change of "
            f"{residual} between the last two iterates, above tol={tol}. On an "
            f"infinite-horizon problem the iteration has not converged; if "
            f"max_iter was set as a finite horizon, this is expected and the "
            f"result is that horizon's solution. Returning the last iterate."
        )
        if raise_on_nonconvergence:
            raise RuntimeError(msg)
        warnings.warn(msg)

    return dr, value_array, policy_array
