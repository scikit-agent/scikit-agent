from HARK.distributions import Normal
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

case_4 = {
    "block": DBlock(
        **{
            "name": "maliar test - non-trivial ergodic states",
            "shocks": {
                "theta": (Normal, {"mu": 0, "sigma": 1}),
                "psi": (Normal, {"mu": 0, "sigma": 1}),
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
