"""
Introduction to Economic Modeling
=================================

This example provides an introduction to the concepts used in economic
modeling with scikit-agent, using only basic Python and matplotlib.
"""

# %%
# Import basic Python libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
# Economic modeling often involves utility functions
# Let's create a simple CRRA (Constant Relative Risk Aversion) utility function
def crra_utility(consumption, gamma=2.0):
    """
    CRRA utility function: u(c) = c^(1-γ) / (1-γ)
    
    Parameters:
    - consumption: consumption level(s)
    - gamma: coefficient of relative risk aversion
    """
    if gamma == 1.0:
        return np.log(consumption)
    else:
        return (consumption**(1 - gamma)) / (1 - gamma)

# %%
# Plot utility for different risk aversion parameters
consumption_levels = np.linspace(0.1, 5.0, 100)
gammas = [0.5, 1.0, 2.0, 4.0]

plt.figure(figsize=(12, 8))

for i, gamma in enumerate(gammas):
    utility_vals = crra_utility(consumption_levels, gamma)
    plt.subplot(2, 2, i+1)
    plt.plot(consumption_levels, utility_vals, 'b-', linewidth=2)
    plt.xlabel('Consumption')
    plt.ylabel('Utility')
    plt.title(f'CRRA Utility (γ = {gamma})')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Economic models often involve optimization
# Let's show a simple consumption-savings problem setup
def budget_constraint(assets, consumption, income=1.0, interest_rate=0.03):
    """
    Budget constraint: a' = (a + y - c) * R
    where a' is next period assets, a is current assets, 
    y is income, c is consumption, R is gross interest rate
    """
    return (assets + income - consumption) * (1 + interest_rate)

# %%
# Visualize the budget constraint
current_assets = 2.0
income = 1.0
interest_rate = 0.03

consumption_choices = np.linspace(0.1, current_assets + income - 0.1, 100)
next_period_assets = [budget_constraint(current_assets, c, income, interest_rate) 
                     for c in consumption_choices]

plt.figure(figsize=(10, 6))
plt.plot(consumption_choices, next_period_assets, 'r-', linewidth=2)
plt.xlabel('Current Consumption')
plt.ylabel('Next Period Assets')
plt.title(f'Budget Constraint (Assets={current_assets}, Income={income}, r={interest_rate})')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.show()

# %%
# Show the trade-off between current and future consumption
print(f"With current assets of {current_assets} and income of {income}:")
print(f"- Maximum consumption today: {current_assets + income:.2f}")
print(f"- If consume everything: next period assets = {budget_constraint(current_assets, current_assets + income, income, interest_rate):.2f}")
print(f"- If consume half: next period assets = {budget_constraint(current_assets, (current_assets + income)/2, income, interest_rate):.2f}")
print(f"- If consume minimum (0.1): next period assets = {budget_constraint(current_assets, 0.1, income, interest_rate):.2f}")

# %%
# This type of analysis forms the foundation for more complex
# dynamic programming models implemented in scikit-agent
print("\nKey concepts in economic modeling:")
print("1. Utility functions capture preferences")
print("2. Budget constraints limit choices") 
print("3. Optimization finds the best choices")
print("4. Dynamic models consider multiple time periods")
print("5. Uncertainty requires probabilistic modeling")