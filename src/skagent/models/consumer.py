import torch
from skagent.distributions import Bernoulli, Lognormal, MeanOneLogNormal
from skagent.model import Control, DBlock, RBlock

"""
Blocks for consumption saving problem (not normalized)
in the style of Carroll's "Solution Methods for Solving
Microeconomic Dynamic Stochastic Optimization Problems"
"""

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,  # note: this can be overridden by the portfolio dynamics
    "Rfree": 1.03,
    "EqP": 0.02,
    "LivPrb": 0.98,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
    "TranShkStd": 0.1,
    "RiskyStd": 0.1,
}

consumption_block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": (Bernoulli, {"p": "LivPrb"}),  # Move to tick or mortality block?
            "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),
        },
        "dynamics": {
            "b": lambda k, R: k * R,
            "y": lambda p, theta: p * theta,
            "m": lambda b, y: b + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: torch.log(c) if CRRA == 1 else c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)

consumption_block_normalized = DBlock(
    **{
        "name": "consumption normalized",
        "shocks": {
            "live": (Bernoulli, {"p": "LivPrb"}),  # Move to tick or mortality block?
            "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),
        },
        "dynamics": {
            "b": lambda k, R, PermGroFac: k * R / PermGroFac,
            "m": lambda b, theta: b + theta,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": "m - c",
            "u": lambda c, CRRA: torch.log(c) if CRRA == 1 else c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)

portfolio_block = DBlock(
    **{
        "name": "portfolio",
        "shocks": {
            "risky_return": (Lognormal, {"mean": "Rfree + EqP", "std": "RiskyStd"})
        },
        "dynamics": {
            "stigma": Control(["a"]),
            "R": lambda stigma, Rfree, risky_return: Rfree
            + (risky_return - Rfree) * stigma,
        },
    }
)

tick_block = DBlock(
    **{
        "name": "tick",
        "dynamics": {
            "k": lambda a: a,
        },
    }
)

cons_problem = RBlock(blocks=[consumption_block_normalized, tick_block])
cons_portfolio_problem = RBlock(
    blocks=[consumption_block_normalized, portfolio_block, tick_block]
)
