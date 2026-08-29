import warnings

import numpy as np

import skagent.ann as ann
import skagent.loss as loss_module


def solve_multiple_controls(
    control_order, bellman_period, givens, calibration=None, epochs=200, loss=None
):
    """
    Solve a block with more than one control by training a policy network
    for each control in turn.

    Each control is given its own :class:`skagent.ann.BlockPolicyNet`. The
    networks are trained one at a time, in the order given by
    ``control_order``, with every network treating the other networks' current
    policies as fixed. A control may appear in ``control_order`` more than once
    to refine it after its neighbours have been updated (e.g.
    ``["c", "d", "c"]``), which is the multi-control analogue of a best-response
    sweep.

    Currently restricted to single-period (non-recurring) reward objectives;
    by default the negative immediate reward
    (:class:`skagent.loss.StaticRewardLoss`) is maximized.

    Parameters
    ----------
    control_order : list of str
        Control symbols, in the order they should be solved. Symbols may repeat
        to schedule additional refinement passes.
    bellman_period : BellmanPeriod
        The model period whose controls are being solved.
    givens : skagent.grid.Grid
        Grid of arrival states and shock realizations to train over.
    calibration : dict, optional
        Deprecated. The period supplied as *bellman_period* already carries the
        calibration the losses are evaluated at, and that is the one used. If
        given, it must agree with the period's; a disagreement raises rather
        than silently evaluating a period's losses at parameters the period was
        not built with.
    epochs : int, optional
        Training epochs per pass. Default is 200.
    loss : type, optional
        A loss-function class with signature
        ``loss(bellman_period, parameters, other_dr)``. Defaults to
        :class:`skagent.loss.StaticRewardLoss`.

    Returns
    -------
    dict
        Mapping from each control symbol to its trained decision rule.

    Raises
    ------
    ValueError
        If *calibration* is given and disagrees with the period's.
    """
    if calibration is not None:
        warnings.warn(
            "calibration is deprecated; the period passed as bellman_period "
            "already carries one, and that is what the losses are evaluated "
            "at.",
            DeprecationWarning,
            stacklevel=2,
        )
        differing = _differing_symbols(calibration, bellman_period.calibration)
        if differing:
            raise ValueError(
                f"calibration disagrees with bellman_period.calibration on "
                f"{differing}; the period is built from a calibration, so pass "
                "only the period"
            )

    # TODO: allow a variable 'loss function generator' once the API has
    # solidified.
    if loss is None:
        loss = loss_module.StaticRewardLoss

    # Control policy networks for each control in the block.
    cpns = {}

    # Invent Policy Neural Networks for each Control variable.
    for control_sym in bellman_period.get_controls():
        cpns[control_sym] = ann.BlockPolicyNet(bellman_period, control_sym=control_sym)

    dict_of_decision_rules = {
        k: v
        for d in [
            cpns[control_sym].get_decision_rule(length=givens.n())
            for control_sym in cpns
        ]
        for k, v in d.items()
    }

    for control_sym in control_order:
        ann.train_block_nn(
            cpns[control_sym],
            givens,
            loss(
                bellman_period,
                bellman_period.calibration,
                dict_of_decision_rules,
            ),
            epochs=epochs,
        )

    return dict_of_decision_rules


def _differing_symbols(first, second):
    """The symbols on which two calibrations disagree, sorted."""
    missing = object()
    return sorted(
        sym
        for sym in set(first) | set(second)
        if not np.array_equal(first.get(sym, missing), second.get(sym, missing))
    )
