from HARK.distributions import Bernoulli
from skagent.model import Control, DBlock

calibration = {
    "DiscFac": 0.96,
    "CRRA": (2.0,),
    "Rfree": 1.03,
    "LivPrb": 0.98,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "shocks": {
            "live": Bernoulli(p=calibration["LivPrb"]),
        },
        "dynamics": {
            "p": lambda PermGroFac, p: PermGroFac * p,
            "r_eff": lambda Rfree, PermGroFac: Rfree / PermGroFac,
            "b_nrm": lambda r_eff, a_nrm: r_eff * a_nrm,
            "m_nrm": lambda b_nrm: b_nrm + 1,
            "c_nrm": Control(["m_nrm"], upper_bound=lambda m_nrm: m_nrm),
            "a_nrm": lambda m_nrm, c_nrm: m_nrm - c_nrm,
            "u": lambda c_nrm, CRRA: c_nrm ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)
