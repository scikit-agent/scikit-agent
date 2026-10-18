from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from skagent import ann
from skagent.algos import maliar, vfi
from skagent.algos.sb3 import PPOAgent
from skagent.algos.tabular import TabularBestResponseSolver
from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Uniform
from skagent.env import GymEnv
from skagent.ground import GroundedBlock
from skagent.grid import Grid
from skagent.models.macid import prisoners_dilemma_block
from skagent.parser import skagent_loader
from skagent.simulation.monte_carlo import Simulator
from skagent.solver import (
    ExactBestResponse,
    ExactStationaryBestResponse,
    NeuralBestResponse,
    StationaryBestResponse,
    project_nash,
    solve_in_order,
    solve_in_relevance_order,
    solve_mean_field,
    solve_symmetric_equilibrium,
)


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

    def test_randomize_must_be_boolean(self):
        with pytest.raises(TypeError, match="randomize must be a bool"):
            Control([], action_space={0, 1}, randomize=1)

    def test_randomization_requires_a_finite_action_space(self):
        with pytest.raises(ValueError, match="requires a finite action_space"):
            Control([], randomize=True)

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


class TestSolverRefusals:
    @pytest.fixture
    def ground(self):
        return GroundedBlock(randomized_control_block(), {})

    def test_method_entry_points_refuse(self, ground):
        with pytest.raises(NotImplementedError, match="randomize=True"):
            TabularBestResponseSolver(ground)
        with pytest.raises(NotImplementedError, match="randomize=True"):
            NeuralBestResponse(ground, panel=None)
        with pytest.raises(NotImplementedError, match="randomize=True"):
            ExactBestResponse(ground, state_grid={})
        with pytest.raises(NotImplementedError, match="randomize=True"):
            ExactStationaryBestResponse(ground, {}, None)

    def test_direct_solver_entry_points_refuse(self, ground):
        period = BellmanPeriod(ground.block, None, {})
        with pytest.raises(NotImplementedError, match="randomize=True"):
            vfi.solve_step(period, lambda states, shocks, parameters: 0, {})
        with pytest.raises(NotImplementedError, match="randomize=True"):
            vfi.solve_bellman(period, {})
        with pytest.raises(NotImplementedError, match="randomize=True"):
            ann.BlockPolicyNet(period)
        with pytest.raises(NotImplementedError, match="randomize=True"):
            maliar.maliar_training_loop(
                period,
                lambda *_: 0,
                Grid.from_dict({"state": np.array([0.0])}),
                {},
            )
        with pytest.raises(NotImplementedError, match="randomize=True"):
            PPOAgent(period, {})
        with pytest.raises(NotImplementedError, match="randomize=True"):
            GymEnv(period, {})

    @pytest.mark.parametrize(
        "schedule",
        [
            lambda method: solve_in_order(method, []),
            lambda method: solve_in_relevance_order(method),
            lambda method: solve_symmetric_equilibrium(method),
        ],
    )
    def test_schedule_entry_points_refuse(self, ground, schedule):
        method = SimpleNamespace(ground=ground)
        with pytest.raises(NotImplementedError, match="randomize=True"):
            schedule(method)

    def test_mean_field_schedule_refuses(self, ground):
        method = SimpleNamespace(ground=ground)
        with pytest.raises(TypeError, match="StationaryBestResponse"):
            solve_mean_field(method, ground, bracket=(0, 1))

        method = StationaryBestResponse()
        method.ground = ground
        with pytest.raises(NotImplementedError, match="randomize=True"):
            solve_mean_field(method, ground, bracket=(0, 1))
