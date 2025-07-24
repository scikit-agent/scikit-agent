#!/usr/bin/env python3
"""
Simplified test for perfect foresight lifetime reward validation.
This is a minimal test to verify the core functionality works.
"""

import unittest
import torch
import numpy as np

# Test seed for reproducibility
TEST_SEED = 42


class TestSimplePerfectForesight(unittest.TestCase):
    """Simplified test suite for perfect foresight lifetime reward validation."""
    
    def setUp(self):
        """Set up test configuration."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
    
    def test_import_maliar(self):
        """Test that we can import the maliar module."""
        try:
            import skagent.algos.maliar as maliar
            self.assertTrue(hasattr(maliar, 'estimate_discounted_lifetime_reward'))
        except ImportError as e:
            self.fail(f"Failed to import maliar module: {e}")
    
    def test_import_perfect_foresight_model(self):
        """Test that we can import perfect foresight models."""
        try:
            import skagent.models.perfect_foresight as pfm
            self.assertTrue(hasattr(pfm, 'block_no_shock'))
            self.assertTrue(hasattr(pfm, 'calibration'))
        except ImportError as e:
            self.fail(f"Failed to import perfect foresight model: {e}")
    
    def test_basic_lifetime_reward_calculation(self):
        """Test basic lifetime reward calculation with perfect foresight."""
        import skagent.algos.maliar as maliar
        import skagent.models.perfect_foresight as pfm
        
        # Use the existing perfect foresight model without shocks
        block = pfm.block_no_shock
        calibration = pfm.calibration
        
        # Simple decision rule: consume 50% of cash-on-hand
        def simple_policy(states, shocks, parameters):
            m = states["a"] * parameters["Rfree"] + states["p"] 
            return {"c": 0.5 * m}
        
        # Test with small time horizon first
        big_t = 3
        initial_states = {"a": 1.0, "p": 1.0}
        
        # This should work without shock space explosion
        reward = maliar.estimate_discounted_lifetime_reward(
            block, 
            calibration["DiscFac"], 
            simple_policy,
            initial_states,
            big_t,
            parameters=calibration
        )
        
        # Basic validation
        self.assertIsNotNone(reward)
        self.assertFalse(torch.isnan(torch.as_tensor(reward)))
        self.assertFalse(torch.isinf(torch.as_tensor(reward)))
        
        # Should be negative (log utility with reasonable consumption)
        self.assertLess(float(reward), 0)
    
    def test_increasing_time_horizons(self):
        """Test that we can handle increasing time horizons without explosion."""
        import skagent.algos.maliar as maliar
        import skagent.models.perfect_foresight as pfm
        
        block = pfm.block_no_shock
        calibration = pfm.calibration
        
        def simple_policy(states, shocks, parameters):
            m = states["a"] * parameters["Rfree"] + states["p"]
            return {"c": 0.5 * m}
        
        initial_states = {"a": 1.0, "p": 1.0}
        
        # Test multiple time horizons - this is the key innovation
        time_horizons = [1, 3, 5, 10, 20]
        rewards = []
        
        for big_t in time_horizons:
            reward = maliar.estimate_discounted_lifetime_reward(
                block, calibration["DiscFac"], simple_policy,
                initial_states, big_t, parameters=calibration
            )
            rewards.append(float(reward))
            
            # Should be finite and reasonable
            self.assertIsNotNone(reward)
            self.assertFalse(torch.isnan(torch.as_tensor(reward)))
            self.assertFalse(torch.isinf(torch.as_tensor(reward)))
        
        # Verify we got reasonable results for all horizons
        # Lifetime rewards should be negative (log utility) and finite
        for i, reward in enumerate(rewards):
            self.assertLess(reward, 0, f"Reward at horizon {time_horizons[i]} should be negative")
            self.assertGreater(reward, -100, f"Reward at horizon {time_horizons[i]} seems too negative")


if __name__ == "__main__":
    unittest.main() 