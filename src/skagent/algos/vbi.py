"""
Value backward induction (VBI).

Derive a decision rule, decision value function, and arrival value function for
a single :class:`~skagent.block.DBlock` stage by backward induction: at each
point of a grid over the decision's information set, solve an exact
:func:`scipy.optimize.minimize` for the control that maximizes the period reward
plus a continuation value.
"""

from skagent.block import DBlock
from inspect import signature
import itertools
import numpy as np
from scipy.optimize import minimize
from typing import Mapping, Sequence
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
    policy_data = grid_to_data_array(state_grid)
    value_data = grid_to_data_array(state_grid)

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
                policy_data.sel(**state_vals).variable.data.put(
                    0, res.x[0]
                )  # will only work for scalar actions
                value_data.sel(**state_vals).variable.data.put(
                    0, srv_function(pre_states, dr_best)
                )
            else:
                print(f"Optimization failure at {state_vals}.")
                print(res)

                dr_best = {c: get_action_rule(res.x[i]) for i, c in enumerate(controls)}

                policy_data.sel(**state_vals).variable.data.put(0, res.x[0])  # ?
                value_data.sel(**state_vals).variable.data.put(
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
        c: ar_from_data(policy_data.transpose(*block.dynamics[c].iset))
        for c in controls
    }

    dec_vf = block.get_decision_value_function(dr_from_data, continuation)
    arr_vf = block.get_arrival_value_function(disc_params, dr_from_data, continuation)

    return dr_from_data, dec_vf, arr_vf
