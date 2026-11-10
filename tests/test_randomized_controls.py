from copy import deepcopy

import numpy as np
import pytest
import yaml

from skagent.algos.tabular import TabularBestResponseSolver
from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import DiscreteDistribution, Uniform
from skagent.env import GymEnv
from skagent.ground import GroundedBlock
from skagent.models.macid import prisoners_dilemma_block
from skagent.parser import skagent_loader
from skagent.simulation.monte_carlo import Simulator
from skagent.solver import ExactBestResponse, project_nash


def randomized_control_block(*, entity=None):
    return DBlock(
        name="randomized_control",
        shocks={},
        dynamics={
            "D": Control([], agent="player", action_space={0, 1}, randomize=True),
            "U": lambda D: D,
        },
        reward={"U": "player"},
        entity=entity,
    )


def simulate(block, policy, seed=0, sample_count=20_000):
    simulator = Simulator(
        {},
        block,
        {"D": policy},
        {},
        seed=seed,
        sample_count=sample_count,
        T_sim=1,
    )
    simulator.initialize_sim()
    return simulator.simulate()


class TestDeclaration:
    def test_python_and_yaml_are_opt_in(self):
        assert Control([]).randomize is False
        assert Control([], action_space={0, 1}, randomize=True).randomize is True

        parsed = yaml.load(
            "D: !Control {iset: [], action_space: [0, 1], randomize: true}",
            Loader=skagent_loader(),
        )
        assert parsed["D"].randomize is True

        parsed_default = yaml.load(
            "D: !Control {iset: [], action_space: [0, 1]}",
            Loader=skagent_loader(),
        )
        assert parsed_default["D"].randomize is False

        parsed_continuous = yaml.load(
            "D: !Control {iset: [], randomize: true}",
            Loader=skagent_loader(),
        )
        assert parsed_continuous["D"].randomize is True
        assert parsed_continuous["D"].action_space is None

    def test_randomize_must_be_boolean(self):
        with pytest.raises(TypeError, match="randomize must be a bool"):
            Control([], action_space={0, 1}, randomize=1)

    def test_randomization_does_not_require_an_action_space(self):
        control = Control([], randomize=True)

        assert control.action_space is None

    def test_nonrandomized_block_construction_is_unchanged(self):
        shocks = {}
        dynamics = {"D": Control([], action_space={0, 1})}

        block = DBlock(shocks=shocks, dynamics=dynamics)

        assert block.shocks is shocks
        assert block.dynamics is dynamics

    def test_generated_noise_is_a_real_information_parent(self):
        declared = Control(
            ["state"], agent="player", action_space={0, 1}, randomize=True
        )
        shocks = {}
        dynamics = {"state": lambda: 0, "D": declared, "U": lambda D: D}
        block = DBlock(
            shocks=shocks,
            dynamics=dynamics,
            reward={"U": "player"},
        )
        control = block.get_control("D")

        assert declared.iset == ["state"]
        assert shocks == {}
        assert dynamics["D"] is declared
        assert control._authored_iset == ["state"]
        assert control.iset == ["state", "D_noise"]
        assert control._randomizer == "D_noise"
        assert set(block.get_shocks()) == {"D_noise"}

    def test_authored_information_cannot_claim_the_reserved_noise_name(self):
        with pytest.raises(ValueError, match="authored information set"):
            DBlock(
                dynamics={
                    "D": Control(["D_noise"], action_space={0, 1}, randomize=True)
                }
            )

    @pytest.mark.parametrize("where", ["shock", "dynamic", "reward"])
    def test_generated_name_collisions_are_rejected(self, where):
        kwargs = {
            "shocks": {},
            "dynamics": {
                "D": Control([], action_space={0, 1}, randomize=True),
            },
            "reward": {},
        }
        if where == "shock":
            kwargs["shocks"]["D_noise"] = Uniform()
        elif where == "dynamic":
            kwargs["dynamics"]["D_noise"] = lambda: 0
        else:
            kwargs["reward"]["D_noise"] = "player"

        with pytest.raises(ValueError, match="D_noise.*already declared"):
            DBlock(**kwargs)

    def test_collision_across_recursive_blocks_is_rejected(self):
        randomized = randomized_control_block()
        authored = DBlock(name="authored", dynamics={"D_noise": lambda: 0})

        with pytest.raises(ValueError, match="both declare 'D_noise'"):
            RBlock(blocks=[randomized, authored])

    @pytest.mark.parametrize("where", ["shock", "dynamic", "reward"])
    def test_deep_replace_cannot_override_generated_noise(self, where):
        block = randomized_control_block()
        updates = {
            "shock": {"shocks": {"D_noise": Uniform()}},
            "dynamic": {"dynamics": {"D_noise": lambda: 0}},
            "reward": {"reward": {"D_noise": "player"}},
        }

        with pytest.raises(ValueError, match="generated randomizer shocks.*D_noise"):
            block.deep_replace(**updates[where])

    def test_deep_replace_rebuilds_noise_when_replacing_a_control(self):
        block = randomized_control_block()

        randomized = block.deep_replace(
            dynamics={
                "state": lambda: 0,
                "D": Control(["state"], action_space={0, 1}, randomize=True),
            }
        )
        deterministic = block.deep_replace(
            dynamics={"D": Control([], action_space={0, 1})}
        )

        assert set(randomized.get_shocks()) == {"D_noise"}
        assert randomized.get_control("D").iset == ["state", "D_noise"]
        assert set(deterministic.get_shocks()) == {"D_noise"}
        assert deterministic.get_control("D").iset == []

        rerandomized = deterministic.deep_replace(
            dynamics={"D": Control([], action_space={0, 1}, randomize=True)}
        )
        assert set(rerandomized.get_shocks()) == {"D_noise"}
        assert rerandomized.get_control("D").iset == ["D_noise"]

    def test_deep_replace_preserves_noise_that_another_rule_reads(self):
        block = randomized_control_block().deep_replace(
            dynamics={"observed_noise": lambda D_noise: D_noise}
        )

        replaced = block.deep_replace(dynamics={"D": Control([], action_space={0, 1})})
        assert set(replaced.get_shocks()) == {"D_noise"}
        assert replaced.transition({"D_noise": 0.25}, {"D": lambda: 1})[
            "observed_noise"
        ] == pytest.approx(0.25)

    def test_deep_replace_preserves_noise_for_a_cross_leaf_reader(self):
        controls = randomized_control_block()
        reader = DBlock(name="reader", dynamics={"observed": lambda D_noise: D_noise})
        RBlock(blocks=[controls, reader])

        deterministic = controls.deep_replace(
            dynamics={"D": Control([], agent="player", action_space={0, 1})}
        )

        replaced = RBlock(blocks=[deterministic, reader])
        assert set(replaced.get_shocks()) == {"D_noise"}
        assert replaced.transition({"D_noise": 0.25}, {"D": lambda: 1})[
            "observed"
        ] == pytest.approx(0.25)

    def test_composition_allows_an_authored_noise_suffixed_arrival(self):
        fixed = DBlock(dynamics={"D": Control([], action_space={0, 1})})
        reader = DBlock(dynamics={"observed": lambda D_noise: D_noise})

        block = RBlock(blocks=[fixed, reader])

        assert "D_noise" in block.get_arrival_states()


class TestSuppliedPolicies:
    def test_binary_frequency_matches_the_policy_threshold(self):
        history = simulate(
            randomized_control_block(),
            lambda noise: (noise < 0.63).astype(float),
        )

        assert set(np.unique(history["D"])) <= {0, 1}
        assert history["D"].mean() == pytest.approx(0.63, abs=0.015)

    def test_wider_action_space_is_supported(self):
        block = DBlock(
            dynamics={"D": Control([], action_space={10, 20, 30}, randomize=True)}
        )
        history = simulate(
            block,
            lambda noise: np.where(noise < 0.2, 10, np.where(noise < 0.7, 20, 30)),
        )

        frequencies = {
            action: np.mean(history["D"] == action) for action in (10, 20, 30)
        }
        assert frequencies[10] == pytest.approx(0.2, abs=0.015)
        assert frequencies[20] == pytest.approx(0.5, abs=0.015)
        assert frequencies[30] == pytest.approx(0.3, abs=0.015)

    def test_continuous_control_can_use_private_noise(self):
        block = DBlock(
            dynamics={"D": Control([], lower_bound=0, upper_bound=1, randomize=True)}
        )

        history = simulate(block, lambda noise: noise, sample_count=200)

        np.testing.assert_array_equal(history["D"], history["D_noise"])

    @pytest.mark.parametrize(("probability", "action"), [(0.0, 0), (1.0, 1)])
    def test_probability_endpoints_are_deterministic(self, probability, action):
        history = simulate(
            randomized_control_block(),
            lambda noise: (noise < probability).astype(float),
            sample_count=200,
        )
        assert np.all(history["D"] == action)

    def test_seed_reproduces_noise_and_actions(self):
        block = randomized_control_block()

        def policy(noise):
            return (noise < 0.5).astype(float)

        first = simulate(block, policy, seed=12, sample_count=100)
        again = simulate(block, policy, seed=12, sample_count=100)
        other = simulate(block, policy, seed=13, sample_count=100)

        np.testing.assert_array_equal(first["D_noise"], again["D_noise"])
        np.testing.assert_array_equal(first["D"], again["D"])
        assert not np.array_equal(first["D_noise"], other["D_noise"])

    def test_invalid_policy_action_uses_existing_validation(self):
        with pytest.raises(ValueError, match="outside its action_space"):
            simulate(randomized_control_block(), lambda noise: noise)


def test_sampled_prisoners_dilemma_payoffs_match_the_relaxed_model():
    def utility_1(D1, D2):
        return 3.0 + 2.0 * D1 - 3.0 * D2 - D1 * D2

    def utility_2(D1, D2):
        return 3.0 - 3.0 * D1 + 2.0 * D2 - D1 * D2

    block = DBlock(
        dynamics={
            "D1": Control([], action_space={0, 1}, randomize=True),
            "D2": Control([], action_space={0, 1}, randomize=True),
            "U1": utility_1,
            "U2": utility_2,
        }
    )
    p1, p2 = 0.35, 0.7
    simulator = Simulator(
        {},
        block,
        {
            "D1": lambda noise: (noise < p1).astype(float),
            "D2": lambda noise: (noise < p2).astype(float),
        },
        {},
        seed=21,
        sample_count=50_000,
        T_sim=1,
    )
    simulator.initialize_sim()
    history = simulator.simulate()
    relaxed = prisoners_dilemma_block.transition(
        {}, {"D1": lambda: p1, "D2": lambda: p2}
    )

    assert history["U1"].mean() == pytest.approx(relaxed["U1"], abs=0.025)
    assert history["U2"].mean() == pytest.approx(relaxed["U2"], abs=0.025)


def test_private_noise_roles_follow_the_existing_causal_criterion():
    block = DBlock(
        shocks={},
        dynamics={
            "D1": Control([], agent="p1", action_space={0, 1}, randomize=True),
            "D2": Control([], agent="p2", action_space={0, 1}, randomize=True),
            "U1": lambda D1, D2: D1 - D2,
            "U2": lambda D1, D2: D2 - D1,
        },
        reward={"U1": "p1", "U2": "p2"},
    )

    assert block.shock_roles({}) == {
        "D1": {"D1_noise": "observed", "D2_noise": "hidden"},
        "D2": {"D1_noise": "hidden", "D2_noise": "observed"},
    }


class TestProjectionAndReconstruction:
    def test_reconstruction_and_discretization_do_not_duplicate_noise(self):
        block = randomized_control_block()
        reconstructed = DBlock(
            name=block.name,
            shocks=dict(block.shocks),
            dynamics=dict(block.dynamics),
            reward=dict(block.reward),
        )
        copied = deepcopy(block)
        discretized = block.discretize({"D_noise": {"N": 3}}, calibration={})

        for candidate in (reconstructed, copied, discretized):
            assert set(candidate.get_shocks()) == {"D_noise"}
            assert candidate.get_control("D").iset == ["D_noise"]
            assert candidate.get_control("D")._randomizer == "D_noise"

    def test_nested_discretization_preserves_the_transformed_block(self):
        nested = RBlock(blocks=[RBlock(blocks=[randomized_control_block()])])

        discretized = nested.discretize({"D_noise": {"N": 3}}, calibration={})

        assert isinstance(discretized.get_shocks()["D_noise"], DiscreteDistribution)

    def test_projection_renames_existing_noise_before_reconstruction(self):
        population = DBlock(
            name="randomized_control",
            dynamics={
                "state": lambda: 0,
                "D": Control(
                    ["state"],
                    agent="player",
                    action_space={0, 1},
                    randomize=True,
                ),
                "U": lambda D: D,
            },
            reward={"U": "player"},
            entity=Entity("person"),
        )
        projection = project_nash(GroundedBlock(population, {"person": 3}))
        projected = projection.block

        assert set(projected.get_shocks()) == {
            "D_noise_actor",
            "D_noise_other",
        }
        assert projected.get_control("D_actor").iset == [
            "state_actor",
            "D_noise_actor",
        ]
        assert projected.get_control("D_other").iset == [
            "state_other",
            "D_noise_other",
        ]
        assert "D_actor_noise" not in projected.get_shocks()
        assert "D_other_noise" not in projected.get_shocks()

        simulator = Simulator(
            projection.calibration,
            projected,
            {
                "D_actor": lambda state, noise: (noise < 0.4).astype(float),
                "D_other": lambda state, noise: (noise < 0.4).astype(float),
            },
            {},
            seed=4,
            sample_count=5,
            T_sim=1,
        )
        simulator.initialize_sim()
        history = simulator.simulate()
        np.testing.assert_array_equal(
            history["D_actor"], history["D_noise_actor"] < 0.4
        )
        np.testing.assert_array_equal(
            history["D_other"], history["D_noise_other"] < 0.4
        )

        rebuilt = deepcopy(projected).discretize(
            {"D_noise_actor": {"N": 3}, "D_noise_other": {"N": 3}},
            calibration={"person_other": 2},
        )
        assert set(rebuilt.get_shocks()) == {
            "D_noise_actor",
            "D_noise_other",
        }

    def test_projected_control_replacement_preserves_cross_leaf_noise(self):
        population = randomized_control_block(entity=Entity("person"))
        projected = project_nash(GroundedBlock(population, {"person": 3})).block
        actor_leaf = next(
            leaf for leaf in projected.iter_dblocks() if "D_actor" in leaf.dynamics
        )

        replaced_leaf = actor_leaf.deep_replace(
            dynamics={
                "D_actor": Control(
                    [],
                    agent="player_actor",
                    action_space={0, 1},
                    randomize=True,
                )
            }
        )
        rebuilt = RBlock(
            name=projected.name,
            blocks=[
                replaced_leaf if leaf is actor_leaf else leaf
                for leaf in projected.blocks
            ],
        )

        assert set(rebuilt.get_shocks()) == {
            "D_noise_actor",
            "D_noise_other",
        }
        assert rebuilt.get_control("D_actor").iset == ["D_noise_actor"]
        assert "D_actor_noise" not in rebuilt.get_shocks()

        deterministic_leaf = actor_leaf.deep_replace(
            dynamics={"D_actor": Control([], agent="player_actor", action_space={0, 1})}
        )
        deterministic = RBlock(
            name=projected.name,
            blocks=[
                deterministic_leaf if leaf is actor_leaf else leaf
                for leaf in projected.blocks
            ],
        )
        assert deterministic.get_control("D_actor").iset == []
        assert "D_noise_actor" in deterministic.get_shocks()

    def test_projection_does_not_restore_noise_to_a_deterministic_control(self):
        population = randomized_control_block(entity=Entity("person")).deep_replace(
            dynamics={"D": Control([], agent="player", action_space={0, 1})}
        )

        projected = project_nash(GroundedBlock(population, {"person": 3})).block

        assert projected.get_control("D_actor").iset == []
        assert projected.get_control("D_other").iset == []
        assert set(projected.get_shocks()) == {
            "D_noise_actor",
            "D_noise_other",
        }

    def test_grounding_rejects_a_bound_control_without_its_noise(self):
        original = randomized_control_block()
        broken = DBlock(dynamics={"D": original.get_control("D")})

        with pytest.raises(ValueError, match="D_noise.*not declared"):
            GroundedBlock(broken, {})


class TestSolverCompatibility:
    def test_exact_solver_can_condition_a_continuous_control_on_noise(self):
        block = DBlock(
            dynamics={
                "D": Control(
                    [], lower_bound=0, upper_bound=1, agent="player", randomize=True
                ),
                "U": lambda D, D_noise: -((D - D_noise) ** 2),
            },
            reward={"U": "player"},
        )
        method = ExactBestResponse(
            GroundedBlock(block, {}),
            {},
            disc_params={"D_noise": {"N": 9}},
        )

        rule = method.best_response("D", method.initial_policies())

        assert rule(np.array([0.25, 0.75])) == pytest.approx([0.25, 0.75], abs=1e-3)

    def test_tabular_solver_reports_its_specific_continuous_cell_limit(self):
        block = randomized_control_block()
        method = TabularBestResponseSolver(
            GroundedBlock(block, {}),
            shock_samples=20,
            max_cells=5,
            rng=np.random.default_rng(2),
        )

        with pytest.raises(ValueError, match="above max_cells=5"):
            method.best_response("D", method.initial_policies())

    def test_gym_environment_routes_only_named_values_to_bounds(self):
        block = DBlock(
            dynamics={
                "state": lambda: 0.75,
                "D": Control(
                    ["state"],
                    lower_bound=0,
                    upper_bound=lambda state: state,
                    agent="player",
                    randomize=True,
                ),
                "U": lambda D: D,
            },
            reward={"U": "player"},
        )
        environment = GymEnv(BellmanPeriod(block, None, {}), {}, seed=3)
        observation, _ = environment.reset()

        assert observation.shape == (2,)
        _, _, _, _, info = environment.step(np.array([0.0], dtype=np.float32))
        assert info["bounds"] == (0.0, 0.75)

    @pytest.mark.parametrize("randomize", [False, True])
    def test_gym_environment_refuses_discrete_controls(self, randomize):
        block = DBlock(
            dynamics={
                "D": Control([], action_space={0, 1}, randomize=randomize),
                "U": lambda D: D,
            },
            reward={"U": "player"},
        )

        with pytest.raises(NotImplementedError, match="discrete action_space"):
            GymEnv(BellmanPeriod(block, None, {}), {})
