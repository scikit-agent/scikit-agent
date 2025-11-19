import skagent.ann as ann


def solve_multiple_controls(
    control_order, bellman_period, givens, calibration, epochs=200, loss=None
):
    """
    Solves a block multiple times, once for each control in control_order.

    Currently restricted to static reward loss.

    TODO: all variable 'loss function generator' once API has solidified.

    Parameters
    ----------
    control_order: list
        List of control symbols in order to be solved
    bellman_period: BellmanPeriod
    """

    if loss is None:
        loss = loss.StaticRewardLoss

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
            loss(  # !!
                bellman_period,
                ["a"],  # !!
                calibration,
                dict_of_decision_rules,
            ),
            epochs=epochs,  # !!
        )

    return dict_of_decision_rules
