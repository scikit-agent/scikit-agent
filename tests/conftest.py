from HARK.distributions import Normal
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
}

case_1 = {
    "block": DBlock(
        **{
            "name": "lr_test_1 - shock",
            "shocks": {
                "theta": (Normal, {"mean": 0, "sigma": 1}),
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
}

lr_test_block_data_2 = {
    "name": "lr_test_2 - hidden shock",
    "shocks": {
        "theta": (Normal, {"mean": 0, "sigma": 1}),
    },
    "dynamics": {
        "c": Control(["a"]),
        "a": lambda a, c, theta: a - c + theta,
        "u": lambda theta, c: -((theta - c) ** 2),
    },
    "reward": {"u": "consumer"},
}
lr_test_block_data_2_optimal_dr = {"c": lambda a: 0}

lr_test_block_data_3 = {
    "name": "lr_test_3 - two shocks, one hidden",
    "shocks": {
        "theta": (Normal, {"mean": 0, "sigma": 1}),
        "psi": (Normal, {"mean": 0, "sigma": 1}),
    },
    "dynamics": {
        "m": lambda a, theta: a + theta,
        "c": Control(["m"]),
        "a": lambda m, c, psi: m - c + psi,
        "u": lambda m, c: -((m - c) ** 2),
    },
    "reward": {"u": "consumer"},
}
lr_test_block_data_3_optimal_dr = {"c": lambda m: m}
