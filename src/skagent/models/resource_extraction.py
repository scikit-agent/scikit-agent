"""
Resource extraction models analyze the optimal management of renewable or depletable resources over time. These models appear in:

    Fisheries management: Determining sustainable harvest rates for fish populations
    Forestry: Optimal timber harvesting schedules
    Environmental economics: Managing renewable natural resources (water, wildlife)
    Energy: Optimal depletion of oil fields and mineral deposits
    Finance: Portfolio liquidation and asset drawdown strategies

The core problem involves balancing immediate extraction (profit now) against preserving the resource stock (profit later), accounting for natural growth dynamics and environmental uncertainty.
"""

from scipy.optimize import fsolve

from skagent.distributions import Normal
from skagent.model import Control, DBlock

calibration = {
    "r": 1.1,  # growth rate
    "p": 5.0,  # price per unit extracted
    "c_param": 10.0,  # cost parameter
    "DiscFac": 0.95,  # discount factor
    "sigma": 0.1,  # standard deviation of growth shock
}

resource_extraction_block = DBlock(
    **{
        "name": "resource_extraction",
        "shocks": {
            "epsilon": (Normal, {"mu": 0.0, "sigma": "sigma"}),
        },
        "dynamics": {
            "u": Control(
                ["x"], lower_bound=0.0, upper_bound=lambda x: x, agent="extractor"
            ),
            "revenue": lambda u, p: p * u,
            "cost": lambda u, c_param: 0.5 * c_param * u**2,
            "profit": lambda revenue, cost: revenue - cost,
            "x_after": lambda x, u: x - u,
            "x_growth": lambda x_after, r: r * x_after,
            "x": lambda x_growth, epsilon: x_growth + epsilon,
        },
        "reward": {"profit": "extractor"},
    }
)


def optimal_extraction_rule(r, p, c_param, DiscFac):
    """
    Compute the optimal linear extraction policy parameters.

    Returns a decision rule function u(x) = alpha * x
    """

    def equations(vars):
        alpha, beta = vars
        eq1 = alpha - p / (c_param + DiscFac * r**2 * beta)
        eq2 = beta - (c_param + DiscFac * r**2 * beta * (1 - alpha) ** 2)
        return [eq1, eq2]

    # Initial guess
    alpha0, beta0 = 0.3, 10.0

    # Solve system
    alpha, beta = fsolve(equations, [alpha0, beta0])

    # Return the decision rule as a function
    def decision_rule(x):
        """Optimal extraction: u = alpha * x"""
        return alpha * x

    return decision_rule


def make_optimal_extraction_decision_function(parameters):
    """Pre-compute alpha for efficiency"""
    r = parameters["r"]
    p = parameters["p"]
    c_param = parameters["c_param"]
    DiscFac = parameters["DiscFac"]

    def equations(vars):
        alpha, beta = vars
        eq1 = alpha - p / (c_param + DiscFac * r**2 * beta)
        eq2 = beta - (c_param + DiscFac * r**2 * beta * (1 - alpha) ** 2)
        return [eq1, eq2]

    alpha, beta = fsolve(equations, [0.3, 10.0])

    def decision_function(states, shocks, parameters):
        x = states["x"]
        return alpha * x

    return decision_function


# Create the optimal decision rule with calibrated parameters
optimal_u = optimal_extraction_rule(
    r=calibration["r"],
    p=calibration["p"],
    c_param=calibration["c_param"],
    DiscFac=calibration["DiscFac"],
)

# Create the decision function
optimal_extraction_df = make_optimal_extraction_decision_function(calibration)
