from skagent.bellman import BellmanPeriod
from skagent.distributions import Normal, Uniform
import skagent.grid as grid
from skagent.model import Control, DBlock

case_0 = {
    "block": DBlock(
        **{
            "name": "very basic case",
            "dynamics": {
                "c": Control(["a"]),
                "u": lambda c: -((c) ** 2),
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: 0},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 2, "count": 21},
        }
    ),
}
case_0["bp"] = BellmanPeriod(case_0["block"], case_0["calibration"])

case_1 = {
    "block": DBlock(
        **{
            "name": "lr_test_1 - shock",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(["a", "theta"]),
                "a": lambda a, c, theta: a - c + theta,
                "u": lambda theta, c: -((theta - c) ** 2),
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a, theta: theta},
    "givens": {
        1: grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 7},
                "theta_0": {"min": -1, "max": 1, "count": 7},
            }
        ),
        2: grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 7},
                "theta_0": {"min": -1, "max": 1, "count": 7},
                "theta_1": {"min": -1, "max": 1, "count": 7},
            }
        ),
    },
}
case_1["bp"] = BellmanPeriod(case_1["block"], case_1["calibration"])

case_2 = {
    "block": DBlock(
        **{
            "name": "lr_test_2 - hidden shock",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(["a"]),
                "a": lambda a, c, theta: a - c + theta,
                "u": lambda theta, c: -((theta - c) ** 2),
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: 0},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 1, "count": 5},
            "theta_0": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_2["bp"] = BellmanPeriod(case_2["block"], case_2["calibration"])

case_3 = {
    "block": DBlock(
        **{
            "name": "lr_test_3 - two shocks, one hidden",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
                "psi": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "m": lambda a, theta: a + theta,
                "c": Control(["m"]),
                "a": lambda m, c, psi: m - c + psi,
                "u": lambda m, c: -((m - c) ** 2),
            },
            "reward": {"u": "consumer"},
        }
    ),
    "optimal_dr": {"c": lambda m: m},
    "calibration": {},
    "givens": {
        1: grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 5},
                "theta_0": {"min": -1, "max": 1, "count": 5},
                "psi_0": {"min": -1, "max": 1, "count": 5},
            }
        ),
        2: grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 5},
                "theta_0": {"min": -1, "max": 1, "count": 5},
                "psi_0": {"min": -1, "max": 1, "count": 3},
                "theta_1": {"min": -1, "max": 1, "count": 5},
                "psi_1": {"min": -1, "max": 1, "count": 3},
            }
        ),
    },
}
case_3["bp"] = BellmanPeriod(case_3["block"], case_3["calibration"])

case_4 = {
    "block": DBlock(
        **{
            "name": "maliar test - non-trivial ergodic states",
            "shocks": {
                "theta": (Uniform, {"low": -1, "high": 1}),
                "psi": (Uniform, {"low": -1, "high": 1}),
            },
            "dynamics": {
                "c": Control(["g", "m"]),
                "a": lambda m, c: m - c,
                "u": lambda a, g: -((a - g) ** 2),
                "m": lambda a, theta: a + theta,
                "g": lambda g, psi: g + psi,
            },
            "reward": {"u": "consumer"},
        }
    ),
    "optimal_dr": {"c": lambda g, m: g - m},
    "calibration": {},
    "givens": {
        2: grid.Grid.from_config(
            {
                "m": {"min": -100, "max": 100, "count": 7},
                "g": {"min": -100, "max": 100, "count": 7},
                "theta_0": {"min": -1, "max": 1, "count": 7},
                "psi_0": {"min": -1, "max": 1, "count": 7},
                "theta_1": {"min": -1, "max": 1, "count": 5},
                "psi_1": {"min": -1, "max": 1, "count": 5},
            }
        ),
    },
}
case_4["bp"] = BellmanPeriod(case_4["block"], case_4["calibration"])

case_5 = {
    "block": DBlock(
        **{
            "name": "double bounded control -- upper bound binds",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(["a"], upper_bound=lambda a: a, lower_bound=lambda a: 0),
                "a": lambda a, c, theta: a - c + 2.72**theta,
                "u": lambda c: c,
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: a},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 1, "count": 5},
            "theta_0": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_5["bp"] = BellmanPeriod(case_5["block"], case_5["calibration"])

case_6 = {
    "block": DBlock(
        **{
            "name": "double bounded control -- lower bound binds",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(
                    ["a"], upper_bound=lambda a: 2 * a, lower_bound=lambda a: a
                ),
                "a": lambda a, c, theta: a + 2.72**theta,
                "u": lambda c: -c,
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: a},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 1, "count": 5},
            "theta_0": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_6["bp"] = BellmanPeriod(case_6["block"], case_6["calibration"])

case_7 = {
    "block": DBlock(
        **{
            "name": "lower bounded control only",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(["a"], lower_bound=lambda a: 1),
                "a": lambda a, c, theta: a + 2.72**theta - c,
                "u": lambda c: -c,
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: 1},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 1, "count": 5},
            "theta_0": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_7["bp"] = BellmanPeriod(case_7["block"], case_7["calibration"])

case_8 = {
    "block": DBlock(
        **{
            "name": "upper bounded control only",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "c": Control(["a"], upper_bound=lambda a: a),
                "a": lambda a, c, theta: a + 2.72**theta - c,
                "u": lambda c: c,
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda a: a},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 1, "count": 5},
            "theta_0": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_8["bp"] = BellmanPeriod(case_8["block"], case_8["calibration"])

case_9 = {
    "block": DBlock(
        **{
            "name": "empty information set",
            "dynamics": {
                "a": lambda a: a,
                "c": Control([]),
                "u": lambda c: -((c - 3) ** 2),
            },
            "reward": {"u": "consumer"},
        }
    ),
    "calibration": {},
    "optimal_dr": {"c": lambda: 3},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": 0, "max": 2, "count": 21},
        }
    ),
}
case_9["bp"] = BellmanPeriod(case_9["block"], case_9["calibration"])

case_10 = {
    "block": DBlock(
        **{
            "name": "multi control case",
            "dynamics": {
                "c": Control(["a"], agent="agent"),
                "d": Control([], agent="agent"),
                "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
            },
            "reward": {"u": "agent"},
        }
    ),
    "calibration": {"k": 3},
    "optimal_dr": {"c": lambda a: a, "d": lambda: 3},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": -2, "max": 2, "count": 11},
        }
    ),
}
case_10["bp"] = BellmanPeriod(case_10["block"], case_10["calibration"])

case_11 = {
    "block": DBlock(
        **{
            "name": "non-trivial continuation value",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
            },
            "dynamics": {
                "u": lambda a, b: -((a - b) ** 2),
                "c": Control(["a", "theta"], agent="agent"),
                "a": lambda a, theta: a + theta,
                "b": lambda c: c,
            },
            "reward": {"u": "agent"},
        }
    ),
    "optimal_dr": {"c": lambda a, theta: a + theta},
    "givens": grid.Grid.from_config(
        {
            "a": {"min": -2, "max": 2, "count": 11},
            "b": {"min": -2, "max": 2, "count": 11},
            "theta_0": {"min": -1, "max": 1, "count": 5},
            "theta_1": {"min": -1, "max": 1, "count": 5},
        }
    ),
}
case_11["bp"] = BellmanPeriod(case_11["block"], {})
