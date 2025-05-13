"""
test.py - Test module for the model_viz visualization module
"""

from skagent.model import DBlock, Control
from model_viz import DBlockVisualizer
import inspect


def create_test_block(with_time_dependencies=True, with_params=True, name="test_block"):
    """
    Factory function to create test blocks with configurable properties.

    Parameters:
    with_time_dependencies -- Whether to include time dependencies in state variables
    with_params -- Whether to include parameter dependencies
    name -- Name of the block

    Returns:
    A configured DBlock instance for testing
    """
    block = DBlock()
    block.name = name

    # Define shock variables with optional parameter dependency
    shock_params = {"mu": 0}
    if with_params:
        shock_params["sigma"] = "param1"  # Parameter dependency
    else:
        shock_params["sigma"] = 0.1  # Fixed value

    block.shocks = {"shock1": (None, shock_params)}

    # Define dynamics with configurable time dependency
    if with_time_dependencies:
        # State variable depends on its previous value (time dependency)
        block.dynamics = {
            "state1": lambda state1, shock1: state1 * 0.9 + shock1,
            "control1": Control(["state1"]),
        }
    else:
        # State variable only depends on shock
        block.dynamics = {
            "state1": lambda shock1: shock1 * 2,
            "control1": Control(["state1"]),
        }

    # Define reward function
    block.reward = {"reward1": lambda state1, control1: state1 + control1}

    # Add optional attribute
    block.default_limit = None

    return block


def test_identify_variables():
    """
    Test whether _identify_variables correctly classifies the block's variables
    into shocks, states, controls, rewards, and parameters.
    """
    block = create_test_block()
    viz = DBlockVisualizer(block)
    viz._identify_variables()
    variables = viz.variables

    # Generate reference sets of variables from block attributes
    reference_shocks = set(block.shocks.keys())

    reference_states = set()
    reference_controls = set()
    for key, rule in block.dynamics.items():
        if isinstance(rule, Control):
            reference_controls.add(key)
        else:
            reference_states.add(key)

    reference_rewards = set(block.reward.keys())

    # Verify variable classification matches expectations
    assert set(variables["shock_vars"]) == reference_shocks, (
        f"reference shock_vars: {reference_shocks}, got: {variables['shock_vars']}"
    )
    print("✓ Shock variables assertion passed")
    assert set(variables["state_vars"]) == reference_states, (
        f"reference state_vars: {reference_states}, got: {variables['state_vars']}"
    )
    print("✓ State variables assertion passed")
    assert set(variables["control_vars"]) == reference_controls, (
        f"reference control_vars: {reference_controls}, got: {variables['control_vars']}"
    )
    print("✓ Control variables assertion passed")
    assert set(variables["reward_vars"]) == reference_rewards, (
        f"reference reward_vars: {reference_rewards}, got: {variables['reward_vars']}"
    )
    print("✓ Reward variables assertion passed")
    assert isinstance(variables["param_vars"], set), "param_vars should be a set type"
    print("✓ Param variables assertion passed")


def test_extract_dependencies():
    """
    Test whether _extract_dependencies correctly extracts dependency relationships
    between variables in the block.
    """
    block = create_test_block(with_time_dependencies=True)
    viz = DBlockVisualizer(block)
    viz._identify_variables()
    viz._extract_dependencies()
    dependencies = viz.dependencies

    # Generate reference dependencies based on block structure
    reference_deps = {}

    for key, rule in block.dynamics.items():
        if isinstance(rule, Control):
            reference_deps[key] = list(rule.iset)
        else:
            sig = inspect.signature(rule)
            # Extract parameters from function signature
            reference_deps[key] = list(sig.parameters.keys())

    for key, rule in block.reward.items():
        sig = inspect.signature(rule)
        reference_deps[key] = list(sig.parameters.keys())

    # Verify extracted dependencies match expectations (order-insensitive)
    for var, reference in reference_deps.items():
        actual = dependencies.get(var, [])
        assert set(actual) == set(reference), (
            f"For variable '{var}', reference dependencies: {reference}, got: {actual}"
        )
        print("✓ Dependencies assertion passed")


def test_identify_time_dependencies():
    """
    Test whether _identify_time_dependencies correctly identifies variables
    that depend on their previous period values.
    """
    # Test with time dependencies
    block_with_time = create_test_block(with_time_dependencies=True)
    viz_with_time = DBlockVisualizer(block_with_time)
    viz_with_time._identify_variables()
    viz_with_time._extract_dependencies()
    viz_with_time._identify_time_dependencies()

    # Test without time dependencies
    block_no_time = create_test_block(with_time_dependencies=False)
    viz_no_time = DBlockVisualizer(block_no_time)
    viz_no_time._identify_variables()
    viz_no_time._extract_dependencies()
    viz_no_time._identify_time_dependencies()

    # Verify time dependency structures are created
    assert hasattr(viz_with_time, "prev_period_vars"), (
        "prev_period_vars should be created"
    )
    print("✓ prev_period_vars structure assertion passed")
    assert hasattr(viz_with_time, "prev_period_deps"), (
        "prev_period_deps should be created"
    )
    print("✓ prev_period_deps structure assertion passed")

    # Verify time dependencies are correctly identified in the first block
    assert "state1" in viz_with_time.prev_period_vars, (
        "Block with time dependencies: state1 should be identified as time-dependent"
    )
    print("✓ time dependency identification assertion passed")

    assert ("state1", "state1") in viz_with_time.prev_period_deps, (
        "Block with time dependencies: (state1, state1) should be in prev_period_deps"
    )
    print("✓ prev_period_deps content assertion passed")

    # Verify no time dependencies are found in the second block
    assert "state1" not in viz_no_time.prev_period_vars, (
        "Block without time dependencies: state1 should not be time-dependent"
    )
    print("✓ no time dependency assertion passed")


def test_extract_formulas():
    """
    Test whether _extract_formulas correctly extracts formula information
    for all model variables and parameters.
    """
    # Example calibration parameters
    calibration = {"param1": 0.05, "r": 0.03}
    block = create_test_block(with_time_dependencies=True)
    viz = DBlockVisualizer(block, calibration=calibration)

    # Run the complete analysis flow
    viz._identify_variables()
    viz._extract_dependencies()
    viz._identify_time_dependencies()
    viz._extract_formulas()
    formulas = viz.formulas

    # reference formula variables include dynamics, rewards, and calibration parameters
    reference_formula_vars = (
        set(block.dynamics.keys()) | set(block.reward.keys()) | set(calibration.keys())
    )
    missing = reference_formula_vars - set(formulas.keys())
    assert not missing, (
        f"The following variables should appear in formulas but are missing: {missing}"
    )
    print("✓ Formula variables completeness assertion passed")

    # Check Control variables have "Control" in their formula
    for var, rule in block.dynamics.items():
        if isinstance(rule, Control) and var in formulas:
            assert "Control" in formulas[var], (
                f"Formula for Control variable {var} should include Control keyword"
            )
            print(f"✓ Control formula assertion for {var} passed")

    # Check calibration parameters have their values in formulas
    for param, value in calibration.items():
        if param in formulas:
            assert str(value) in formulas[param], (
                f"Formula for parameter {param} should include its value {value}"
            )
            print(f"✓ Parameter value assertion for {param} passed")

    # Check time dependencies are marked in formulas
    for var in viz.prev_period_vars:
        if var in formulas:
            assert var in formulas[var], (
                f"Formula for time dependent variable {var} should include self-reference"
            )
            print(f"✓ Time dependency in formula assertion for {var} passed")


def test_agent_attribution():
    """
    Test whether get_agent_for_variable correctly maps variables to their associated agents.
    """
    block = create_test_block()
    agent_attribution = {
        "agent1": ["state1", "shock1"],
        "agent2": ["control1", "reward1"],
    }
    viz = DBlockVisualizer(block, agent_attribution=agent_attribution)

    # Test regular variables
    for agent, vars_list in agent_attribution.items():
        for var in vars_list:
            assert viz.get_agent_for_variable(var) == agent, (
                f"Variable {var} should be attributed to {agent}"
            )
            print(
                f"✓ Regular variable attribution assertion for {var} -> {agent} passed"
            )
    # Test time-dependent notation
    assert viz.get_agent_for_variable("state1*") == "agent1", (
        "Time-dependent notation state1* should be attributed to agent1"
    )
    print("✓ Time-dependent notation attribution assertion passed")

    # Test internal _prev notation
    assert viz.get_agent_for_variable("state1_prev") == "agent1", (
        "Internal notation state1_prev should be attributed to agent1"
    )
    print("✓ Internal notation attribution assertion passed")

    # Test unattributed variable
    assert viz.get_agent_for_variable("unknown") == "other", (
        "Unattributed variable should be mapped to 'other'"
    )
    print("✓ Unattributed variable attribution assertion passed")


def test_analyze():
    """
    Test whether the analyze method correctly executes the full analysis pipeline.
    """
    block = create_test_block()
    calibration = {"param1": 0.05, "r": 0.03}
    agent_attribution = {
        "agent1": ["state1", "shock1"],
        "agent2": ["control1", "reward1"],
    }

    viz = DBlockVisualizer(block, agent_attribution, calibration)

    # Run analyze
    result = viz.analyze()

    # Check method chaining
    assert result is viz, "analyze() should return self for method chaining"
    print("✓ Method chaining assertion passed")

    # Check that all data structures are populated
    assert viz.variables, "variables should be populated"
    print("✓ Variables population assertion passed")

    assert viz.dependencies, "dependencies should be populated"
    print("✓ Dependencies population assertion passed")

    assert viz.formulas, "formulas should be populated"
    print("✓ Formulas population assertion passed")

    assert hasattr(viz, "prev_period_vars"), "prev_period_vars should be created"
    print("✓ prev_period_vars structure assertion passed")

    # Check specific results to ensure full pipeline ran
    assert set(viz.variables["shock_vars"]) == set(block.shocks.keys()), (
        "Shock variables not correctly analyzed"
    )
    print("✓ Shock variables analysis assertion passed")

    assert "state1" in viz.prev_period_vars, "Time dependencies not correctly analyzed"
    print("✓ Time dependencies analysis assertion passed")

    assert "state1" in viz.dependencies, "Dependencies not correctly analyzed"
    print("✓ Dependencies content assertion passed")

    assert "state1" in viz.formulas, "Formulas not correctly analyzed"
    print("✓ Formulas content assertion passed")


if __name__ == "__main__":
    test_identify_variables()
    test_extract_dependencies()
    test_identify_time_dependencies()
    test_extract_formulas()
    test_agent_attribution()
    test_analyze()
    print("All tests passed!")
