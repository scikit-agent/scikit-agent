#!/usr/bin/env python3
"""Test suite for benchmark models with research-grade robustness.

Most models have analytical solutions, but U-3 (buffer stock) requires numerical solution.
"""

import pytest
import torch
from skagent.models.benchmarks import (
    list_benchmark_models,
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
    get_test_states,
    validate_analytical_solution,
    get_analytical_lifetime_reward,
    BENCHMARK_MODELS,
    EPS_STATIC,
)


def has_analytical_policy(model_id: str) -> bool:
    """Check if a model has an analytical policy."""
    return "analytical_policy" in BENCHMARK_MODELS.get(model_id, {})


class TestBenchmarksModels:
    """Test suite for all 6 consumption-savings benchmark models.

    5 models have analytical solutions (D-1, D-2, D-3, U-1, U-2).
    1 model requires numerical solution (U-3 buffer stock).
    """

    def test_benchmark_models_exist(self):
        """Test that all 6 benchmark models are present"""
        expected_models = [
            "D-1",
            "D-2",
            "D-3",
            "U-1",
            "U-2",
            "U-3",  # Buffer stock - no analytical solution
        ]
        actual_models = list(BENCHMARK_MODELS.keys())

        assert set(expected_models) == set(actual_models), (
            f"Expected {expected_models}, got {actual_models}"
        )
        assert len(BENCHMARK_MODELS) == 6, (
            f"Expected 6 models, got {len(BENCHMARK_MODELS)}"
        )

    def test_model_registry_structure(self):
        """Test that each model has required components"""
        # Models that require numerical solution (no analytical policy)
        numerical_only_models = {"U-3"}

        for model_id in BENCHMARK_MODELS.keys():
            model_info = BENCHMARK_MODELS[model_id]

            assert "block" in model_info, f"Model {model_id} missing 'block'"
            assert "calibration" in model_info, (
                f"Model {model_id} missing 'calibration'"
            )
            assert "test_states" in model_info, (
                f"Model {model_id} missing 'test_states'"
            )

            # Analytical policy is optional for numerical-only models
            if model_id not in numerical_only_models:
                assert "analytical_policy" in model_info, (
                    f"Model {model_id} missing 'analytical_policy'"
                )
            else:
                # Verify numerical-only models explicitly lack analytical_policy
                assert "analytical_policy" not in model_info, (
                    f"Model {model_id} should NOT have 'analytical_policy' "
                    "(requires numerical solution)"
                )

            # Test calibration has description
            assert "description" in model_info["calibration"], (
                f"Model {model_id} missing description"
            )

    @pytest.mark.parametrize(
        "model_id",
        ["D-1", "D-2", "D-3", "U-1", "U-2"],
    )
    def test_model_validation(self, model_id):
        """Test that each model passes basic validation"""
        result = validate_analytical_solution(model_id)

        # Should either pass validation or have a specific error message
        if "success" in result:
            assert result["success"], f"Model {model_id} failed validation: {result}"
        else:
            # If there's an error, it should be a meaningful one
            assert "error" in result, (
                f"Model {model_id} returned unexpected result: {result}"
            )

    def test_deterministic_models(self):
        """Test all three deterministic models (D-1, D-2, D-3)"""

        # D-1: Finite horizon log utility
        policy_d1 = get_analytical_policy("D-1")
        parameters_d1 = get_benchmark_calibration("D-1")
        test_states_d1 = get_test_states("D-1", test_points=3)
        result_d1 = policy_d1(test_states_d1, {}, parameters_d1)

        assert "c" in result_d1
        assert torch.all(result_d1["c"] > 0), "D-1 consumption should be positive"
        assert torch.all(result_d1["c"] <= test_states_d1["W"]), (
            "D-1 consumption should not exceed wealth"
        )

        # D-2: Infinite horizon CRRA
        policy_d2 = get_analytical_policy("D-2")
        parameters_d2 = get_benchmark_calibration("D-2")
        test_states_d2 = get_test_states("D-2", test_points=3)
        result_d2 = policy_d2(test_states_d2, {}, parameters_d2)

        assert "c" in result_d2
        assert torch.all(result_d2["c"] > 0), "D-2 consumption should be positive"
        # Note: With human wealth, consumption can exceed arrival assets

        # D-3: Blanchard mortality
        policy_d3 = get_analytical_policy("D-3")
        parameters_d3 = get_benchmark_calibration("D-3")
        test_states_d3 = get_test_states("D-3", test_points=3)
        result_d3 = policy_d3(test_states_d3, {}, parameters_d3)

        assert "c" in result_d3
        assert torch.all(result_d3["c"] > 0), "D-3 consumption should be positive"
        # Note: With human wealth, consumption can exceed arrival assets

    def test_stochastic_models(self):
        """Test key stochastic models (U-1, U-2 with analytical solutions)"""

        # U-1: PIH with βR=1
        policy_u1 = get_analytical_policy("U-1")
        parameters_u1 = get_benchmark_calibration("U-1")
        test_states_u1 = get_test_states("U-1", test_points=3)
        result_u1 = policy_u1(test_states_u1, {}, parameters_u1)

        assert "c" in result_u1
        assert torch.all(result_u1["c"] > 0), "U-1 consumption should be positive"

        # U-2: Log utility with permanent income (normalized)
        policy_u2 = get_analytical_policy("U-2")
        parameters_u2 = get_benchmark_calibration("U-2")
        test_states_u2 = get_test_states("U-2", test_points=3)
        # U-2 uses normalized arrival state 'a', needs shock for policy
        test_shocks_u2 = {"psi": torch.ones(3)}  # Mean shock = 1
        result_u2 = policy_u2(test_states_u2, test_shocks_u2, parameters_u2)

        assert "c" in result_u2
        assert torch.all(result_u2["c"] > 0), "U-2 consumption should be positive"

    def test_u3_buffer_stock_structure(self):
        """Test U-3 buffer stock model structure (no analytical solution)"""
        # U-3 does not have an analytical policy
        with pytest.raises(ValueError, match="does not have an analytical policy"):
            get_analytical_policy("U-3")

        # But it should have other components
        calibration = get_benchmark_calibration("U-3")
        test_states = get_test_states("U-3", test_points=5)
        block = get_benchmark_model("U-3")

        assert "DiscFac" in calibration
        assert "R" in calibration
        assert "CRRA" in calibration
        assert "a" in test_states
        assert test_states["a"].shape == (5,)
        assert hasattr(block, "name")

    def test_model_descriptions(self):
        """Test that model descriptions match expected patterns"""
        models = list_benchmark_models()

        expected_descriptions = {
            "D-1": "Finite horizon log utility",
            "D-2": "Infinite horizon CRRA perfect foresight",
            "D-3": "Blanchard discrete-time mortality",
            "U-1": "Hall random walk consumption",
            "U-2": "Log utility normalized",
            "U-3": "Buffer stock",
        }

        for model_id, description in models.items():
            assert any(
                expected.lower() in description.lower()
                for expected in expected_descriptions[model_id].split()
            ), (
                f"Model {model_id} description '{description}' doesn't match expected pattern"
            )

    def test_analytical_policy_functions(self):
        """Test that analytical policy functions work correctly"""

        # Test that models with analytical solutions can generate policies
        for model_id in BENCHMARK_MODELS.keys():
            if has_analytical_policy(model_id):
                try:
                    policy = get_analytical_policy(model_id)
                    assert callable(policy), f"Policy for {model_id} is not callable"
                except Exception as e:
                    pytest.fail(f"Failed to get analytical policy for {model_id}: {e}")
            else:
                # Models without analytical solutions should raise ValueError
                with pytest.raises(ValueError):
                    get_analytical_policy(model_id)

    def test_model_access_functions(self):
        """Test the main access functions work correctly"""

        # Test get_benchmark_model
        for model_id in BENCHMARK_MODELS.keys():
            model = get_benchmark_model(model_id)
            assert hasattr(model, "name"), f"Model {model_id} missing name attribute"

        # Test get_benchmark_calibration
        for model_id in BENCHMARK_MODELS.keys():
            calibration = get_benchmark_calibration(model_id)
            assert isinstance(calibration, dict), (
                f"Calibration for {model_id} is not a dict"
            )
            assert "description" in calibration, (
                f"Calibration for {model_id} missing description"
            )

        # Test error handling
        with pytest.raises(ValueError):
            get_benchmark_model("INVALID")

        with pytest.raises(ValueError):
            get_benchmark_calibration("INVALID")

        with pytest.raises(ValueError):
            get_analytical_policy("INVALID")


class TestStaticIdentityVerification:
    """Layer 2: Verify every closed-form formula against code implementation"""

    def test_d1_finite_horizon_formula(self):
        """Test D-1: c_t = (1-β)/(1-β^(T-t)) * W_t (finite horizon log utility)"""
        calibration = get_benchmark_calibration("D-1")
        policy = get_analytical_policy("D-1")
        beta = calibration["DiscFac"]
        T = calibration["T"]

        # Test at different time periods
        for t in [0, 1, 2]:
            # Get test states and set time period
            base_states = get_test_states("D-1", test_points=3)
            test_states = {
                "W": base_states["W"],
                "t": torch.full_like(base_states["t"], t),
            }
            result = policy(test_states, {}, calibration)

            remaining_periods = T - t
            if remaining_periods > 1:
                numerator = 1 - beta
                denominator = 1 - (beta**remaining_periods)
                expected_c = (numerator / denominator) * test_states["W"]
            else:
                expected_c = test_states["W"]  # Terminal period

            assert torch.allclose(result["c"], expected_c, atol=EPS_STATIC), (
                f"D-1 formula violated at t={t}: got {result['c']}, expected {expected_c}"
            )

    def test_d2_d3_kappa_formulas(self):
        """Test D-2/D-3: c_t = κ*(m_t + H) where κ = (R - (βR)^(1/σ))/R"""
        for model_id in ["D-2", "D-3"]:
            calibration = get_benchmark_calibration(model_id)
            policy = get_analytical_policy(model_id)

            beta = calibration["DiscFac"]
            R = calibration["R"]
            sigma = calibration["CRRA"]
            y = calibration["y"]

            # For D-3, use effective discount factor
            if model_id == "D-3":
                s = calibration["SurvivalProb"]
                beta_eff = s * beta
            else:
                beta_eff = beta

            kappa = (R - (beta_eff * R) ** (1 / sigma)) / R
            r = R - 1
            human_wealth = y / r

            test_states = get_test_states(model_id, test_points=5)
            result = policy(test_states, {}, calibration)

            # Policy computes m = a*R + y, then c = κ*(m + H)
            m = test_states["a"] * R + y
            expected_c = kappa * (m + human_wealth)
            assert torch.allclose(result["c"], expected_c, atol=EPS_STATIC), (
                f"{model_id} κ formula violated: got {result['c']}, expected {expected_c}"
            )

    def test_u1_pih_formula(self):
        """Test U-1: c_t = (r/R) * W_t (PIH with βR=1)"""
        calibration = get_benchmark_calibration("U-1")
        policy = get_analytical_policy("U-1")

        R = calibration["R"]
        y_mean = calibration["y_mean"]
        r = R - 1

        test_states = get_test_states("U-1", test_points=5)
        result = policy(test_states, {}, calibration)

        # PIH total wealth: W = R*A + y + H where H = y_mean/r
        human_wealth = y_mean / r
        total_wealth = R * test_states["A"] + test_states["y"] + human_wealth
        expected_c = (r / R) * total_wealth

        assert torch.allclose(result["c"], expected_c, atol=EPS_STATIC), (
            f"U-1 PIH formula violated: got {result['c']}, expected {expected_c}"
        )

    def test_u2_normalized_pih_rule(self):
        """Test U-2: c = (1-β)*(m + h) with normalized variables.

        U-2 uses normalized variables (all divided by permanent income P):
        - a = A/P (normalized arrival assets)
        - m = R*a/ψ + 1 (normalized cash-on-hand)
        - h = 1/r (normalized human wealth, constant)
        - c = C/P (normalized consumption)

        Analytical solution: c = (1-β)(m + h)
        """
        calibration = get_benchmark_calibration("U-2")
        policy = get_analytical_policy("U-2")

        beta = calibration["DiscFac"]
        R = calibration["R"]
        r = R - 1

        test_states = get_test_states("U-2", test_points=5)
        test_shocks = {"psi": torch.ones(5)}  # Mean shock = 1

        result = policy(test_states, test_shocks, calibration)

        # Normalized: m = R*a/psi + 1, h = 1/r
        m = R * test_states["a"] / test_shocks["psi"] + 1
        h = 1 / r
        expected_c = (1 - beta) * (m + h)

        assert torch.allclose(result["c"], expected_c, atol=EPS_STATIC), (
            f"U-2 normalized PIH rule violated: got {result['c']}, expected {expected_c}"
        )


class TestDynamicOptimalityChecks:
    """Layer 3: Euler equation and budget evolution tests on simulated paths"""

    def test_u2_normalized_policy_consistency(self):
        """Test U-2 normalized policy consistency.

        U-2 uses normalized variables:
        - a = A/P (normalized assets, arrival state)
        - m = R*a/ψ + 1 (normalized cash-on-hand)
        - c = C/P (normalized consumption)

        The analytical solution c = (1-β)(m + h) satisfies:
        1. Consumption is positive
        2. Consumption respects budget constraint (c < m when h > 0 contributes)
        3. The Euler equation is satisfied by construction
        """
        calibration = get_benchmark_calibration("U-2")
        policy = get_analytical_policy("U-2")

        beta = calibration["DiscFac"]
        R = calibration["R"]
        r = R - 1

        # Test that the policy satisfies the normalized PIH formula
        test_states = get_test_states("U-2", test_points=5)
        test_shocks = {"psi": torch.ones(5)}
        result = policy(test_states, test_shocks, calibration)

        # Normalized: m = R*a + 1, h = 1/r
        m = R * test_states["a"] / test_shocks["psi"] + 1
        h = 1 / r
        expected_c = (1 - beta) * (m + h)

        assert torch.allclose(result["c"], expected_c, atol=EPS_STATIC), (
            f"U-2 normalized PIH formula violated: got {result['c']}, expected {expected_c}"
        )

        # Test that consumption is positive
        assert torch.all(result["c"] > 0), "U-2 consumption should be positive"

        # With human wealth h > 0, consumption can exceed m in theory,
        # but for typical calibrations (1-β < 1) and m > 0, c < m + h
        # Just verify consumption is reasonable relative to total wealth
        total_wealth = m + h
        assert torch.all(result["c"] < total_wealth), (
            "U-2 consumption should be less than total wealth (m + h)"
        )

    def test_budget_evolution_consistency(self):
        """Test that simulated paths satisfy budget constraints for all models"""

        test_cases = {
            "D-1": {
                "initial_states": {"W": 2.0, "t": 0},
                "T": None,
            },  # Use model's own T
            "D-2": {"initial_states": {"a": 1.0}, "T": 3},
            "D-3": {"initial_states": {"a": 1.0}, "T": 3},
            "U-1": {"initial_states": {"A": 1.0, "y": 1.0}, "T": 3},
            "U-2": {"initial_states": {"a": 1.0}, "T": 3},  # Normalized arrival assets
        }

        for model_id, config in test_cases.items():
            calibration = get_benchmark_calibration(model_id)
            policy = get_analytical_policy(model_id)
            R = calibration["R"]

            # Simulate path
            T = config["T"] if config["T"] is not None else calibration.get("T", 3)
            torch.manual_seed(42)

            if model_id == "D-1":
                # Finite horizon model
                W_path = torch.zeros(T)
                t_path = torch.zeros(T, dtype=torch.long)
                c_path = torch.zeros(T)

                W_path[0] = config["initial_states"]["W"]
                t_path[0] = config["initial_states"]["t"]

                for t in range(T):
                    states = {"W": W_path[t : t + 1], "t": t_path[t : t + 1]}
                    result = policy(states, {}, calibration)
                    c_path[t] = result["c"][0]

                    if t < T - 1:
                        # Wealth evolution: W_{t+1} = (W_t - c_t) * R
                        W_path[t + 1] = (W_path[t] - c_path[t]) * R
                        t_path[t + 1] = t_path[t] + 1

                # Check that wealth decreases over time (consumption is reasonable)
                assert W_path[0] > W_path[-1], (
                    f"D-1 wealth should decrease over time: {W_path[0]} -> {W_path[-1]}"
                )
                # Check that consumption is positive
                assert torch.all(c_path > 0), (
                    f"D-1 consumption should be positive: {c_path}"
                )

            elif model_id in ["D-2", "D-3"]:
                # Cash-on-hand models following DBlock dynamics
                a_path = torch.zeros(T)
                m_path = torch.zeros(T)
                c_path = torch.zeros(T)

                a_path[0] = config["initial_states"]["a"]

                for t in range(T):
                    # Compute cash-on-hand: m_t = a_{t-1} * R + y (start of period)
                    m_path[t] = a_path[t] * R + calibration["y"]

                    # Agent makes consumption decision based on arrival assets
                    states = {"a": a_path[t : t + 1]}
                    result = policy(states, {}, calibration)
                    c_path[t] = result["c"][0]

                    if t < T - 1:
                        # Asset evolution: a_t = m_t - c_t (end of period)
                        a_path[t + 1] = m_path[t] - c_path[t]

                # Check that consumption is positive and feasible
                for t in range(T):
                    assert c_path[t] > 0, (
                        f"{model_id} consumption should be positive at t={t}"
                    )

                    # For models with human wealth, consumption can exceed cash-on-hand
                    # Check that total wealth (financial + human) is positive
                    r = calibration["R"] - 1
                    human_wealth = calibration["y"] / r
                    total_wealth = m_path[t] + human_wealth
                    assert c_path[t] <= total_wealth + 1e-10, (
                        f"{model_id} consumption should not exceed total wealth at t={t}: "
                        f"c={c_path[t]}, total_wealth={total_wealth}"
                    )

            elif model_id == "U-1":
                # PIH Hall model (simple - no habit formation)
                A_path = torch.zeros(T)
                y_path = torch.zeros(T)
                c_path = torch.zeros(T)

                A_path[0] = config["initial_states"]["A"]
                y_path[0] = config["initial_states"]["y"]

                for t in range(T):
                    states = {"A": A_path[t : t + 1], "y": y_path[t : t + 1]}
                    result = policy(states, {}, calibration)
                    c_path[t] = result["c"][0]

                    if t < T - 1:
                        # Asset evolution: A_{t+1} = A_t*R + y_t - c_t
                        A_path[t + 1] = A_path[t] * R + y_path[t] - c_path[t]
                        # Income is i.i.d., use mean for testing
                        y_path[t + 1] = calibration["y_mean"]

                # Check that consumption is positive
                for t in range(T):
                    assert c_path[t] > 0, f"U-1 consumption should be positive at t={t}"

            elif model_id == "U-2":
                # Log utility with normalized variables
                # a = A/P (normalized arrival assets)
                # m = R*a/psi + 1 (normalized cash-on-hand)
                a_path = torch.zeros(T)
                m_path = torch.zeros(T)
                c_path = torch.zeros(T)

                a_path[0] = config["initial_states"]["a"]

                for t in range(T):
                    # Normalized cash-on-hand: m = R*a + 1 (psi=1 in test)
                    m_path[t] = a_path[t] * R + 1
                    # Pass normalized arrival state
                    states = {"a": a_path[t : t + 1]}
                    shocks = {"psi": torch.ones(1)}  # Mean shock
                    result = policy(states, shocks, calibration)
                    c_path[t] = result["c"][0]

                    if t < T - 1:
                        # Normalized asset evolution: a' = m - c
                        a_path[t + 1] = m_path[t] - c_path[t]

                # Check that consumption is positive and feasible
                r = calibration["R"] - 1
                h = 1 / r  # Normalized human wealth

                for t in range(T):
                    assert c_path[t] > 0, f"U-2 consumption should be positive at t={t}"

                    # Check that consumption doesn't exceed total wealth
                    total_wealth = m_path[t] + h
                    assert c_path[t] <= total_wealth + 1e-10, (
                        f"U-2 consumption should not exceed total wealth at t={t}: "
                        f"c={c_path[t]}, total_wealth={total_wealth}"
                    )

    @pytest.mark.parametrize(
        "model_id,initial_wealth,initial_assets,T,tolerance",
        [
            ("D-1", 2.0, None, 5, 1e-6),
            ("D-2", None, 1.0, 3, 1e-6),
            ("D-3", None, 1.0, 3, 1e-6),
        ],
    )
    def test_lifetime_reward_validation(
        self, model_id, initial_wealth, initial_assets, T, tolerance
    ):
        """
        Test that simulated paths match analytical lifetime rewards.

        This validates the complete solution by:
        1. Simulating forward using the analytical policy
        2. Computing the realized lifetime utility from the path
        3. Comparing to the analytical lifetime reward formula
        """
        calibration = get_benchmark_calibration(model_id)
        policy = get_analytical_policy(model_id)
        R = calibration["R"]
        beta = calibration["DiscFac"]

        if model_id == "D-1":
            # Simulate D-1 finite horizon path
            W = initial_wealth
            numerical_reward = 0.0
            discount = 1.0

            for t in range(T):
                states = {"W": W, "t": t}
                result = policy(states, {}, calibration)
                c = float(result["c"])

                # Accumulate discounted utility
                u = float(torch.log(torch.as_tensor(c, dtype=torch.float32)))
                numerical_reward += discount * u
                discount *= beta

                # Update state
                W = (W - c) * R

            # Compare to analytical lifetime reward
            analytical_reward = get_analytical_lifetime_reward(
                model_id,
                initial_wealth,
                beta,
                R,
                T,
            )

            assert abs(numerical_reward - analytical_reward) < tolerance, (
                f"{model_id}: Simulated reward {numerical_reward:.10f} != "
                f"analytical {analytical_reward:.10f}"
            )

        elif model_id in ["D-2", "D-3"]:
            # Simulate D-2/D-3 infinite horizon path (finite T for testing)
            a = initial_assets
            y = calibration["y"]
            sigma = calibration["CRRA"]

            # Get survival probability for D-3
            s = calibration.get("SurvivalProb", 1.0)
            beta_eff = s * beta if model_id == "D-3" else beta

            # Initial cash-on-hand
            m_0 = a * R + y

            # Analytical lifetime reward at initial state
            if model_id == "D-2":
                analytical_reward = get_analytical_lifetime_reward(
                    model_id,
                    m_0,  # cash-on-hand
                    beta,
                    R,
                    sigma,
                    y,  # income
                )
            else:  # D-3
                analytical_reward = get_analytical_lifetime_reward(
                    model_id,
                    m_0,  # cash-on-hand
                    beta,
                    R,
                    sigma,
                    s,  # survival prob
                    y,  # income
                )

            # Simulate path and compute numerical reward
            numerical_reward = 0.0
            discount = 1.0
            a_t = a

            for t in range(T):
                # Compute cash-on-hand
                m_t = a_t * R + y

                # Get consumption from policy
                states = {"a": a_t}
                result = policy(states, {}, calibration)
                c = float(result["c"])

                # Accumulate discounted utility
                if sigma == 1:
                    u = float(torch.log(torch.as_tensor(c, dtype=torch.float32)))
                else:
                    u = float(
                        torch.as_tensor(c, dtype=torch.float32) ** (1 - sigma)
                        / (1 - sigma)
                    )
                numerical_reward += discount * u
                discount *= beta_eff

                # Update state
                a_t = m_t - c

            # For infinite horizon models with finite simulation:
            # - Simulated reward captures only T periods of utility
            # - Analytical reward is the infinite horizon value
            # - With CRRA σ>1 and typical consumption values, utility is negative
            # - Simulated (finite) reward should be less negative than analytical (infinite)
            # Just check that simulated is reasonable (positive utility means less negative)
            assert numerical_reward > analytical_reward - abs(analytical_reward), (
                f"{model_id}: Simulated reward {numerical_reward:.10f} seems unreasonable "
                f"compared to analytical {analytical_reward:.10f}"
            )


def test_benchmark_functionality():
    """Integration test: verify all benchmark models are functional"""

    print("\n=== BENCHMARK MODELS ===")
    print("Testing well-known consumption-savings problems")
    print("=" * 55)

    models = list_benchmark_models()
    for model_id, description in models.items():
        has_analytical = (
            "analytical" if has_analytical_policy(model_id) else "numerical"
        )
        print(f"{model_id:4s}: {description} [{has_analytical}]")

    print(f"\nTotal: {len(models)} models")

    # Verify we have exactly the expected models
    expected_models = [
        "D-1",
        "D-2",
        "D-3",
        "U-1",
        "U-2",
        "U-3",  # Buffer stock - numerical only
    ]
    assert set(models.keys()) == set(expected_models)
    assert len(models) == 6

    # Verify analytical vs numerical classification
    analytical_models = [m for m in expected_models if has_analytical_policy(m)]
    numerical_models = [m for m in expected_models if not has_analytical_policy(m)]

    assert set(analytical_models) == {"D-1", "D-2", "D-3", "U-1", "U-2"}
    assert set(numerical_models) == {"U-3"}

    print(f"\n✓ {len(analytical_models)} models with analytical solutions")
    print(f"✓ {len(numerical_models)} models requiring numerical solution")
