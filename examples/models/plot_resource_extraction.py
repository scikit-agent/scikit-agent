"""
############################
Resource Extraction Model
############################

The resource extraction problem models the optimal management of a renewable
or depletable resource (fishery, forest, mineral deposit, etc.).
The decision-maker must balance immediate profits from extraction against
preserving the resource stock for future use. Under certain formulations with
stochastic growth, this problem admits analytical or semi-analytical solutions

The model involves a growing resource with stochastic shocks, a controllable
extraction rate, and reward function that combines revenue and extraction costs.
The planner agent must balance present profit with future resource availability.

Model Structure
==================

- **State Variable:** :math:`x`: Resource stock level
- **Control Variable:** :math:`u`: Extraction rate (constrained between 0 and current stock $x$)

**Dynamics**

The resource stock evolves as:

 .. math::

    x_{t+1} = r(x_t - u_t) + \epsilon_t

where :math:`r > 1` is the growth rate and :math:`\epsilon_t` is a random shock.

**Reward:**

Single-period profit combines revenue and extraction costs:

 .. math::

    \mathrm{profit} = p \cdot u - \\frac{c}{2} u^2

The linear revenue term :math:`p \cdot u` represents market value, while the quadratic cost :math:`\\frac{c}{2}u^2` reflects increasing marginal extraction costs (it becomes harder and more expensive to extract at higher rates).

**Parameters:**

- :math:`r`: Growth rate (e.g., 1.1 = 10% growth per period)
- :math:`p`: Price per unit extracted
- :math:`c`: Cost parameter (controls extraction cost curvature)
- :math:`\gamma` (`DiscFac`): Discount factor for future rewards
- :math:`\sigma`: Standard deviation of random growth shocks

"""

import matplotlib.pyplot as plt
import skagent as ska
from skagent.distributions import Normal
import skagent.models.resource_extraction as rex

# %%
# Model Inspection
# -------------------
#
# First, let's load the predefined model elements and inspect them.

# %%
# Step 1: Show Model Parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("Model Calibration:")
for param, value in rex.calibration.items():
    print(f"  {param}: {value}")

# %%
# Step 2: Inspect the Resource Extraction Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rex.resource_extraction_block.display_formulas()

# %%
# Step 3: Visualize the Resource Extraction Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rex.resource_extraction_block.display(rex.calibration)

# %%
# Step 4: Load Optimal Decision Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This model has a closed-form optimal policy: the optimal extraction is **linear in the stock**:
#
# .. math::
#
#     u^* = \alpha \cdot x
#
# The optimal extraction rate $\alpha$ is determined by solving a system of two equations involving
# the model parameters. A method for solving this system is provided as ``optimal_extraction_rule``.
#
# This analytical solution makes the model ideal for validating other policy
# learning algorithms—we can compare learned policies against the known optimum.
#

optimal_u = rex.optimal_extraction_rule(
    r=rex.calibration["r"],
    p=rex.calibration["p"],
    c_param=rex.calibration["c_param"],
    DiscFac=rex.calibration["DiscFac"],
)


# Wrap rules in the format expected by simulator
decision_rule = {"u": optimal_u}

# %%
# Step 5: Run Monte Carlo Simulation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Initial conditions (must be distributions, not scalar values)
initial_conditions = {
    "x": Normal(mu=0.0, sigma=rex.calibration["sigma"]),
}

# Create and run simulator
simulator = ska.MonteCarloSimulator(
    calibration=rex.calibration,
    block=rex.resource_extraction_block,
    dr=decision_rule,
    initial=initial_conditions,
    agent_count=1000,  # Simulate 5000 agents
    T_sim=100,  # For 100 periods
    seed=42,  # For reproducibility
)

# Run the simulation
print("Running simulation...")
simulator.initialize_sim()  # Initialize simulation variables
simulator.simulate()

print("✓ Simulation completed successfully")

# %%
# Step 6: Plot Simulation Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plt.figure()
plt.plot(simulator.history["x"].mean())
