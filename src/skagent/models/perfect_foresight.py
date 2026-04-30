r"""
Perfect-foresight consumption-savings with stochastic survival and
permanent income growth.

Block representation of the canonical perfect-foresight problem with
i.i.d. survival probability :math:`s \in (0, 1)` and gross permanent
income growth :math:`G`. The agent solves

.. math::
    \max \, \sum_{t=0}^{\infty} (s\beta)^t \,
    \frac{c_t^{\,1-\sigma}}{1 - \sigma}
    \quad \text{s.t.} \quad
    m_{t+1} = R \, (m_t - c_t) + y_{t+1},
    \qquad y_{t+1} = G \, P_t,

with permanent income growing as :math:`P_{t+1} = G\, P_t`. Two
conditions are needed for the closed form: mortality-adjusted
return-impatience :math:`(s\beta R)^{1/\sigma} < R` (so consumption does
not explode), and :math:`R > G` (so human wealth is finite). Under both,
the consumption rule is linear in total wealth :math:`W_t = m_t + H_t`,
with human wealth :math:`H_t = G\, P_t / (R - G)` and MPC
:math:`\kappa_s = (R - (s\beta R)^{1/\sigma})/R`. The companion module
:mod:`skagent.models.perfect_foresight_normalized` solves the same
problem in variables divided by :math:`P_t`, which collapses the state
space and is the form used in the buffer-stock literature.

Notes
-----
The math above uses :math:`R`, :math:`G`, and :math:`s` for the gross
return, permanent income growth, and survival probability; the
corresponding block parameter keys are ``Rfree``, ``PermGroFac``, and
``LivPrb``.

References
----------
Carroll, C.D. (2024). *Solution Methods for Solving Microeconomic Dynamic
Stochastic Optimization Problems*.
https://llorracc.github.io/SolvingMicroDSOPs/
"""

from skagent.distributions import Bernoulli
from skagent.block import Control, DBlock

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "Rfree": 1.03,
    "LivPrb": 0.98,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": Bernoulli(p=calibration["LivPrb"]),
        },
        "dynamics": {
            "y": lambda p: p,
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)

block_no_shock = DBlock(
    **{
        "name": "consumption",
        "dynamics": {
            "y": lambda p: p,
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)
