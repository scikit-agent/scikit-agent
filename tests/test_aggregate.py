"""How a derivative passes through an aggregate of a population's own decisions.

The economy is Aiyagari's with the wage scaled by ``nu``. At ``nu = 1`` the
Cobb-Douglas identity ``K R_K + W_K = 0`` makes the price effects of all
households cancel when the average endowment is one, which would hide a return
factor computed under the wrong concept; ``nu = 1.2`` breaks it.
"""

import numpy as np
import pytest
import torch

import skagent.models.aiyagari as aiyagari
import skagent.models.cournot as cournot
from skagent import ann, loss
from skagent.bellman import BellmanPeriod, estimate_bellman_foc_residual
from skagent.block import DBlock, Entity, RBlock
from skagent.ground import GroundedBlock
from skagent.grid import Grid
from skagent.solver import ExactBestResponse, project_nash, solve_symmetric_equilibrium

ALPHA, DELTA, NU, N = aiyagari.CAPITAL_SHARE, aiyagari.DEPRECIATION, 1.2, 64


def economy(reduce=lambda a: a.mean()):
    market = DBlock(
        name="market",
        dynamics={
            "K": reduce,
            "R": lambda K, alpha, delta: alpha * K ** (alpha - 1) - delta,
            "W": lambda K, alpha, nu: nu * (1 - alpha) * K**alpha,
        },
    )
    households = RBlock(
        name="households",
        entity=Entity("household"),
        blocks=[aiyagari.household_block],
    )
    return RBlock(name="economy", blocks=[market, households])


def period(aggregate, block=None):
    calibration = aiyagari.aiyagari_calibration(size=N, sigma=0.3)
    calibration["nu"] = NU
    return BellmanPeriod(
        block if block is not None else economy(),
        "DiscFac",
        calibration,
        rng=np.random.default_rng(0),
        aggregate=aggregate,
    )


def panel(shape=(N,)):
    g = torch.Generator().manual_seed(0)
    assets = 5.45 * torch.exp(
        0.4 * torch.randn(shape, generator=g, dtype=torch.float64)
    )
    endowment = torch.exp(0.3 * torch.randn(shape, generator=g, dtype=torch.float64))
    # Mean one along the entity axis, so that only nu breaks the identity.
    return assets, endowment / endowment.mean(-1, keepdim=True)


def prices(a):
    K = a.mean(-1, keepdim=True)
    R = ALPHA * K ** (ALPHA - 1) - DELTA
    R_K = ALPHA * (ALPHA - 1) * K ** (ALPHA - 2)
    W_K = NU * ALPHA * (1 - ALPHA) * K ** (ALPHA - 1)
    return K, R, R_K, W_K


def return_factor(bp, a, theta):
    """dz/da per household: the factor the Euler residual uses."""
    a = a.clone().requires_grad_(True)
    grads = bp.grad_pre_state_function(
        {"a": a}, {"a": a}, shocks={"theta": theta}, control_sym="c"
    )
    return grads["z"]["a"].detach()


class TestTheEulerReturnFactor:
    def test_detach_gives_each_household_the_price_taking_return(self):
        a, theta = panel()
        _, R, _, _ = prices(a)
        factor = return_factor(period("detach"), a, theta)
        torch.testing.assert_close(factor, (1 + R).expand_as(a))

    def test_own_share_adds_each_household_its_own_price_effect(self):
        a, theta = panel()
        _, R, R_K, W_K = prices(a)
        nash = 1 + R + (theta * W_K + a * R_K) / N
        torch.testing.assert_close(return_factor(period("own_share"), a, theta), nash)

    def test_through_sums_every_households_price_effect(self):
        # The derivative as the block writes it, which is the planner's channel
        # on a reward objective and no concept on a residual one.
        a, theta = panel()
        K, R, R_K, W_K = prices(a)
        summed = 1 + R + W_K * theta.mean() + R_K * K
        torch.testing.assert_close(
            return_factor(period("through"), a, theta), summed.expand_as(a)
        )

    def test_several_economies_in_one_batch_each_see_their_own_prices(self):
        # Economies on a leading axis, households last: each economy's
        # aggregate is reduced over its own households and no one else's.
        bp = period("own_share", economy(lambda a: a.mean(-1, keepdim=True)))
        a, theta = panel((3, N))
        _, R, R_K, W_K = prices(a)
        nash = 1 + R + (theta * W_K + a * R_K) / N
        torch.testing.assert_close(return_factor(bp, a, theta), nash)


class TestTheBellmanFirstOrderCondition:
    def test_detach_gives_the_price_taking_marginal_value(self):
        # With V(z) = log z, the price-taking dV(z')/dc is -(1 + R') / z'. The
        # cross terms of 'through' are weighted by each household's V', so the
        # identity would not rescue it even at nu = 1.
        bp = period("detach")
        a, theta = panel()
        theta_next = theta.flip(0)
        share = 0.3

        def df(states, shocks, parameters):
            z = bp.compute_pre_state("c", states, shocks=shocks, parameters=parameters)
            return {"c": share * z["z"]}

        def vf(states, shocks, parameters):
            z = bp.compute_pre_state("c", states, shocks=shocks, parameters=parameters)
            return torch.log(z["z"])

        residual = estimate_bellman_foc_residual(
            bp, vf, df, {"a": a}, {"theta_0": theta, "theta_1": theta_next}
        )["c"]

        z = bp.compute_pre_state("c", {"a": a}, shocks={"theta": theta})["z"]
        c, a_next = share * z, (1 - share) * z
        _, R_next, _, _ = prices(a_next)
        z_next = theta_next * NU * (1 - ALPHA) * a_next.mean() ** ALPHA
        z_next = z_next + (1 + R_next) * a_next
        dv_dc = (residual - c**-2.0) / bp.calibration["DiscFac"]
        torch.testing.assert_close(dv_dc, -(1 + R_next) / z_next)


class TestValuesAreUnchanged:
    @pytest.mark.parametrize("aggregate", ["detach", "own_share", "through"])
    def test_every_mode_computes_the_same_period(self, aggregate):
        a, theta = panel()
        c = 0.3 * a
        reference = period("through").post_function(
            {"a": a}, {"c": c}, shocks={"theta": theta}
        )
        post = period(aggregate).post_function(
            {"a": a.clone().requires_grad_(True)}, {"c": c}, shocks={"theta": theta}
        )
        for sym in ("z", "u", "a"):
            torch.testing.assert_close(post[sym].detach(), reference[sym])


class TestRefusals:
    def test_a_derivative_through_an_unnamed_aggregate_is_refused(self):
        bp = period(None)
        a, theta = panel()
        with pytest.raises(ValueError, match="aggregate="):
            return_factor(bp, a, theta)

    def test_a_loss_on_an_unnamed_aggregate_is_refused_when_built(self):
        bp = BellmanPeriod(
            cournot.cournot_block, None, dict(cournot.collusion_calibration())
        )
        with pytest.raises(ValueError, match="aggregate="):
            loss.StaticRewardLoss(bp)

    def test_a_residual_loss_through_the_aggregate_is_refused(self):
        with pytest.raises(ValueError, match="no equilibrium condition"):
            loss.EulerEquationLoss(period("through"), constrained=True)

    def test_the_planner_is_reached_by_the_lifetime_reward(self):
        loss.EstimatedDiscountedLifetimeRewardLoss(period("through"), big_t=2)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="aggregate must be one of"):
            period("mean_field")


class TestBlocksWithoutTheQuestion:
    def test_a_household_alone_needs_no_mode(self):
        calibration = aiyagari.aiyagari_calibration(sigma=0.3)
        calibration.update(R=0.04, W=1.2)
        bp = BellmanPeriod(aiyagari.household_block, "DiscFac", calibration)
        assert bp.aggregated_decisions == {}
        loss.EulerEquationLoss(bp, constrained=True)

    def test_a_projection_has_settled_the_question(self):
        projected = project_nash(
            GroundedBlock(cournot.cournot_block, cournot.collusion_calibration())
        )
        bp = BellmanPeriod(projected.block, None, projected.calibration)
        assert bp.aggregated_decisions == {}


COST = 4.0


def cournot_quantity(aggregate, size):
    """One shared rule trained on the unprojected market, one firm per sample."""
    torch.manual_seed(0)
    bp = BellmanPeriod(
        cournot.cournot_block,
        None,
        cournot.collusion_calibration(size=size, cost=COST),
        aggregate=aggregate,
    )
    net = ann.BlockPolicyNet(bp, control_sym="q", width=16)
    panel = Grid.from_dict({"c": torch.full((size,), COST)})
    ann.train_block_nn(net, panel, loss.StaticRewardLoss(bp, agent="firm"), epochs=600)
    quantity = net.get_decision_rule(length=size)["q"](panel["c"])
    return quantity.detach().mean().item()


class TestCournotUnderEachConcept:
    """Each mode reaches its concept's closed form on the raw Cournot market.

    The population is trained as one shared rule, with no projection and no
    outer schedule: the market's price responds in value to every firm's
    quantity, so the panel clears the market while the mode decides what each
    firm believes its own quantity does to the price.
    """

    @pytest.mark.parametrize("size", [3, 4])
    def test_detach_reaches_the_price_taking_quantity(self, size):
        # Price equals marginal cost: A - b Q = c.
        competitive = (cournot.A - COST) / cournot.B
        assert cournot_quantity("detach", size) == pytest.approx(competitive, abs=1e-3)

    @pytest.mark.parametrize("size", [3, 4])
    def test_own_share_reaches_the_nash_quantity(self, size):
        # At four firms undamped best-response iteration diverges; training one
        # shared rule solves the first-order condition directly and does not.
        nash = cournot.nash_quantity(size=size, cost=COST)
        assert cournot_quantity("own_share", size) == pytest.approx(nash, abs=1e-3)

    @pytest.mark.parametrize("size", [3, 4])
    def test_through_reaches_the_joint_monopoly_quantity(self, size):
        monopoly = cournot.monopoly_quantity(cost=COST)
        assert cournot_quantity("through", size) == pytest.approx(monopoly, abs=1e-3)

    def test_own_share_agrees_with_the_projection(self):
        # Two realizations of Nash among N: the derivative rule on the population
        # and the actor-and-others projection solved by best-response iteration.
        size = 3
        projected = project_nash(
            GroundedBlock(
                cournot.cournot_block, cournot.collusion_calibration(size, COST)
            )
        )
        method = ExactBestResponse(
            projected,
            {"c_actor": np.array([COST])},
            scope={**projected.calibration, "c_other": COST},
        )
        rule, info = solve_symmetric_equilibrium(
            method, damping=2.0 / (size + 1), max_iterations=12
        )
        assert info["converged"]
        by_projection = float(np.atleast_1d(rule(np.array([COST]))).ravel()[0])
        assert cournot_quantity("own_share", size) == pytest.approx(
            by_projection, abs=0.02
        )
