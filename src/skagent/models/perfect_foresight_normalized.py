r"""
Perfect-foresight consumption-savings in normalized variables.

The same problem as :mod:`skagent.models.perfect_foresight`, but with
every level variable divided by permanent income :math:`P_t`. Lowercase
ratios :math:`m = M/P`, :math:`c = C/P`, :math:`a = A/P` evolve via the
effective return :math:`R_{\text{eff}} = R / G`,

.. math::
    b_{t+1} = R_{\text{eff}} \, a_t,
    \qquad
    m_{t+1} = b_{t+1} + 1,
    \qquad
    a_{t+1} = m_{t+1} - c_{t+1},

with normalized income identically equal to one. Normalization reduces
the state space from :math:`(M, P)` to the single ratio :math:`m`, which
matters both for analytical tractability and for neural-network solvers,
where the network learns a one-dimensional function :math:`c(m)` instead
of a two-dimensional :math:`c(M, P)`.

The closed-form normalized policy is :math:`c_t = \kappa_s\, (m_t + h)`
with :math:`\kappa_s = (R - (s\beta R)^{1/\sigma})/R` and
:math:`h = 1 / (R_{\text{eff}} - 1)`. Although the normalized Bellman
discounts future utility by :math:`s\beta G^{1-\sigma}` rather than
:math:`s\beta`, the algebra collapses
:math:`(s\beta G^{1-\sigma} R_{\text{eff}})^{1/\sigma}` to
:math:`(s\beta R)^{1/\sigma}/G`, so the level and normalized MPCs agree.

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
