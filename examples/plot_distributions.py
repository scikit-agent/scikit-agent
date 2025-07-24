"""
Working with Distributions
==========================

This example shows how to use the various distribution classes available
in scikit-agent for modeling economic shocks and uncertainties.
"""

# %%
# Import the distribution classes
from skagent.distributions import Bernoulli, Lognormal, MeanOneLogNormal
import numpy as np
import matplotlib.pyplot as plt

# %%
# Create a Bernoulli distribution for survival probability
survival_prob = Bernoulli(p=0.98)
print(f"Survival distribution with p={survival_prob.p}")
print(f"Mean: {survival_prob.mean}")
print(f"Std: {survival_prob.std}")

# %%
# Sample from the distribution
np.random.seed(42)  # For reproducible results
survival_samples = survival_prob.draw(n=1000)
survival_rate = np.mean(survival_samples)
print(f"Simulated survival rate: {survival_rate:.3f}")

# %%
# Create and sample from a mean-one lognormal distribution
# This is commonly used for transitory income shocks
income_shock = MeanOneLogNormal(sigma=0.1)
print(f"Income shock distribution with sigma={income_shock.sigma_param}")
print(f"Mean: {income_shock.mean}")
print(f"Std: {income_shock.std}")

# %%
# Sample income shocks and plot histogram
income_samples = income_shock.draw(n=10000)
plt.figure(figsize=(10, 6))
plt.hist(income_samples, bins=50, alpha=0.7, density=True)
plt.axvline(np.mean(income_samples), color='red', linestyle='--', 
           label=f'Sample Mean: {np.mean(income_samples):.3f}')
plt.axvline(1.0, color='green', linestyle='--', 
           label='Expected Mean: 1.0')
plt.xlabel('Income Shock Value')
plt.ylabel('Density')
plt.title('Distribution of Transitory Income Shocks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Verify the mean-one property
print(f"Sample mean: {np.mean(income_samples):.4f}")
print(f"Sample std: {np.std(income_samples):.4f}")
print(f"Expected std: {income_shock.std:.4f}")

# %%
# Compare different sigma values for mean-one lognormal
sigmas = [0.05, 0.1, 0.2, 0.3]
plt.figure(figsize=(12, 8))

for i, sigma in enumerate(sigmas):
    shock_dist = MeanOneLogNormal(sigma=sigma)
    samples = shock_dist.draw(n=5000)
    
    plt.subplot(2, 2, i+1)
    plt.hist(samples, bins=50, alpha=0.7, density=True)
    plt.axvline(np.mean(samples), color='red', linestyle='--', 
               label=f'Sample Mean: {np.mean(samples):.3f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Sigma = {sigma}, Std = {shock_dist.std:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()