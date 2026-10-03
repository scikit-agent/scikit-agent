import numpy as np
import pytest
import yaml

from skagent.block import Control, DBlock, Entity, RBlock, simulate_dynamics
from skagent.distributions import Uniform
from skagent.model_analyzer import ModelAnalyzer
from skagent.model_visualizer import ModelVisualizer
from skagent.models.macid import iterated_prisoners_dilemma_block
from skagent.parser import skagent_loader
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator
from skagent.solver import project


def randomizing_block(*, entity=None):
    return DBlock(
        name="coin",
        dynamics={"D": Control([], randomizes=True)},
        entity=entity,
    )


def test_control_randomizes_is_opt_in_in_python_and_yaml():
    assert Control([]).randomizes is False
    assert Control([], randomizes=True).randomizes is True
    parsed = yaml.load(
        "D: !Control {iset: [], randomizes: true}", Loader=skagent_loader()
    )
    assert parsed["D"].randomizes is True
    defaulted = yaml.load("D: !Control {iset: []}", Loader=skagent_loader())
    assert defaulted["D"].randomizes is False


def test_randomizer_is_a_model_shock_and_causal_parent_not_information():
    block = DBlock(
        dynamics={"D": Control([], randomizes=True), "U": lambda D: D},
        reward={"U": "global"},
    )
    assert set(block.get_vars()) == {"u_D", "D", "U"}
    assert block.get_dynamics()["D"].iset == []

    analysis = ModelAnalyzer(block, {}).analyze()
    assert analysis.node_meta["u_D"]["kind"] == "shock"
    assert ("u_D", "D") in analysis.edges["shock"]
    scim = analysis.influence_graph()
    assert scim.graph.has_edge("u_D", "D")
    assert "u_D" in scim.parents("D")
    assert "u_D" not in scim.information("D")
    assert block.shock_roles()["D"]["u_D"] == "hidden"
    randomizer_node = ModelVisualizer(analysis.to_dict())._make_node("u_D")
    assert randomizer_node.get_attributes()["peripheries"] == "2"


def test_u_prefixed_authored_observation_is_not_hidden_for_a_regular_control():
    block = DBlock(
        shocks={"u_D": (Uniform, {"low": 0.0, "high": 1.0})},
        dynamics={"D": Control(["u_D"]), "U": lambda D: D},
        reward={"U": "global"},
    )
    scim = ModelAnalyzer(block, {}).analyze().influence_graph()
    assert "u_D" in scim.information("D")


@pytest.mark.parametrize(
    ("probability", "draw", "action"),
    [(0.0, 0.0, 0), (1.0, 0.999, 1), (0.4, 0.3, 1), (0.4, 0.4, 0)],
)
def test_fixed_draws_realize_binary_actions(probability, draw, action):
    vals = simulate_dynamics(
        {"D": Control([], randomizes=True)},
        {"u_D": draw},
        {"D": lambda: probability},
    )
    assert vals["D"] == action


def test_randomizing_transition_requires_its_causal_randomizer():
    with pytest.raises(KeyError, match="u_D"):
        simulate_dynamics(
            {"D": Control([], randomizes=True)},
            {},
            {"D": lambda: 0.5},
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_probabilities_raise(probability):
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        simulate_dynamics(
            {"D": Control([], randomizes=True)},
            {"u_D": 0.5},
            {"D": lambda: probability},
        )


def test_randomization_is_elementwise_for_sampled_arrays():
    vals = simulate_dynamics(
        {"D": Control([], randomizes=True)},
        {"u_D": np.array([0.1, 0.7, 0.2])},
        {"D": lambda: np.array([0.2, 0.6, 0.2])},
    )
    np.testing.assert_array_equal(vals["D"], [1, 0, 0])


def test_generated_name_collisions_are_rejected_locally_and_recursively():
    with pytest.raises(ValueError, match="u_D.*already declared"):
        DBlock(
            shocks={"u_D": object()},
            dynamics={"D": Control([], randomizes=True)},
        )

    with pytest.raises(ValueError, match="u_D.*already declared"):
        DBlock(
            dynamics={"D": Control([], randomizes=True)},
            reward={"u_D": "player"},
        )

    randomizer = randomizing_block()
    other = DBlock(dynamics={"u_D": lambda: 0.0})
    with pytest.raises(ValueError, match="both declare 'u_D'"):
        RBlock(blocks=[randomizer, other])

    with pytest.raises(ValueError, match="information set"):
        DBlock(dynamics={"D": Control(["u_D"], randomizes=True)})


def test_generated_randomizer_participates_in_block_discretization():
    block = randomizing_block().discretize({"u_D": {"N": 3}})
    randomizer = block.get_shocks()["u_D"]
    assert not isinstance(randomizer, tuple)
    assert len(randomizer.points) == 3

    copied = block.deep_replace(name="copied")
    assert len(copied.get_shocks()["u_D"].points) == 3

    nested = RBlock(blocks=[RBlock(blocks=[randomizing_block()])])
    nested_randomizer = nested.discretize({"u_D": {"N": 4}}).get_shocks()["u_D"]
    assert len(nested_randomizer.points) == 4


def simulate_coin(seed, *, entity=None, calibration=None):
    block = randomizing_block(entity=entity)
    sim = Simulator(
        calibration or {},
        block,
        {"D": lambda: 0.5},
        {},
        seed=seed,
        sample_count=20,
        T_sim=3,
    )
    sim.initialize_sim()
    return sim.simulate()


def test_seeded_simulation_records_reproducible_draws_and_actions():
    first = simulate_coin(12)
    again = simulate_coin(12)
    other = simulate_coin(13)
    np.testing.assert_array_equal(first["u_D"], again["u_D"])
    np.testing.assert_array_equal(first["D"], first["u_D"] < 0.5)
    assert set(first) == {"u_D", "D"}
    assert set(np.unique(first["D"])) <= {0, 1}
    assert not np.array_equal(first["u_D"], other["u_D"])


def test_sampled_payoffs_match_multilinear_expected_payoffs():
    p1, p2 = 0.35, 0.7
    game = DBlock(
        dynamics={
            "D1": Control([], randomizes=True),
            "D2": Control([], randomizes=True),
            "U1": lambda D1, D2: 3.0 + 2.0 * D1 - 3.0 * D2 - D1 * D2,
            "U2": lambda D1, D2: 3.0 - 3.0 * D1 + 2.0 * D2 - D1 * D2,
        }
    )
    sim = Simulator(
        {},
        game,
        {"D1": lambda: p1, "D2": lambda: p2},
        {},
        seed=21,
        sample_count=50_000,
        T_sim=1,
    )
    sim.initialize_sim()
    history = sim.simulate()
    assert history["U1"].mean() == pytest.approx(
        3.0 + 2.0 * p1 - 3.0 * p2 - p1 * p2, abs=0.025
    )
    assert history["U2"].mean() == pytest.approx(
        3.0 - 3.0 * p1 + 2.0 * p2 - p1 * p2, abs=0.025
    )


def test_entity_randomizers_and_actions_have_entity_shape():
    history = simulate_coin(2, entity=Entity("player"), calibration={"player": 4})
    assert history["u_D"].shape == (3, 20, 4)
    assert history["D"].shape == (3, 20, 4)
    np.testing.assert_array_equal(history["D"], history["u_D"] < 0.5)


def test_entity_projection_preserves_randomization_under_copied_names():
    block = DBlock(
        dynamics={"D": Control([], agent="player", randomizes=True)},
        reward={"U": "player"},
        entity=Entity("person"),
    )
    projected = project(GroundedBlock(block, {"person": 2})).block
    assert projected.get_controls()["D_actor"].randomizes is True
    assert projected.get_controls()["D_other"].randomizes is True
    assert {"u_D_actor", "u_D_other"} <= set(projected.get_shocks())


def test_iterated_prisoners_dilemma_records_binary_actions_and_randomizers():
    sim = Simulator(
        {},
        iterated_prisoners_dilemma_block,
        {
            "D1": lambda previous_D1, previous_D2: previous_D2,
            "D2": lambda previous_D1, previous_D2: previous_D1,
        },
        {},
        seed=4,
        sample_count=8,
        T_sim=3,
    )
    # Tit-for-Tat at probabilities 0 and 1 remains deterministic.
    sim.vars_now["previous_D1"] = np.zeros(8)
    sim.vars_now["previous_D2"] = np.ones(8)
    sim.initialize_sim()
    history = sim.simulate()
    assert {"u_D1", "u_D2", "D1", "D2"} <= set(history)
    np.testing.assert_array_equal(history["D1"][0], np.ones(8))
    np.testing.assert_array_equal(history["D2"][0], np.zeros(8))


def test_iterated_game_keeps_lagged_observations_separate_from_execution_noise():
    block = iterated_prisoners_dilemma_block
    scim = ModelAnalyzer(block, {}).analyze().influence_graph(dynamic=True)
    assert set(scim.information("D1")) == {"previous_D1*", "previous_D2*"}
    assert set(scim.information("D2")) == {"previous_D1*", "previous_D2*"}
    roles = block.shock_roles()
    assert roles["D1"]["u_D1"] == "hidden"
    assert roles["D2"]["u_D2"] == "hidden"
