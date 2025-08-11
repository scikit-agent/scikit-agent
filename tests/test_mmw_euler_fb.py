import os
import torch
import numpy as np

from skagent.model import DBlock, Control
from skagent.ann import BlockPolicyNet, aggregate_net_loss, train_block_policy_nn
from skagent.algos.maliar import (
    get_euler_fb_loss,
    prepare_aio_training_inputs,
)


TEST_SEED = 20250811


def euler_residual(states_t, controls_t, shocks, params, decision_function):
    w_t = states_t["w"]
    z_t = controls_t["zeta"]
    h_t = controls_t.get("h", torch.ones_like(z_t))

    r_next = (
        states_t.get("r", torch.zeros_like(w_t)) * params["rho_r"] + shocks["eps_r"]
    )
    d_next = (
        states_t.get("delta", torch.zeros_like(w_t)) * params["rho_delta"]
        + shocks["eps_delta"]
    )
    p_next = (
        states_t.get("p", torch.zeros_like(w_t)) * params["rho_p"] + shocks["eps_p"]
    )
    q_next = (
        states_t.get("q", torch.zeros_like(w_t)) * params["rho_q"] + shocks["eps_q"]
    )

    w_next = torch.exp(p_next) * torch.exp(q_next) + (w_t - z_t * w_t) * params[
        "rbar"
    ] * torch.exp(r_next)

    states_next = {"r": r_next, "delta": d_next, "p": p_next, "q": q_next, "w": w_next}
    controls_next = decision_function(states_next, {}, params)
    z_next = controls_next["zeta"]
    c_next = z_next * w_next

    c_t = torch.clamp(z_t * w_t, min=1e-8)
    c_next_safe = torch.clamp(c_next, min=1e-8)
    gamma = params["CRRA"]
    mu_ratio = (
        (1.0 / c_next_safe) / (1.0 / c_t)
        if gamma == 1
        else (c_next_safe / c_t) ** (-gamma)
    )

    euler_term = (
        params["DiscFac"]
        * torch.exp(d_next - states_t.get("delta", torch.zeros_like(w_t)))
        * mu_ratio
        * params["rbar"]
        * torch.exp(r_next)
    )
    return euler_term - h_t


def fb_residual(states_t, controls_t, shocks, params, decision_function=None):
    zeta = controls_t["zeta"]
    h = controls_t.get("h", torch.ones_like(zeta))
    a = 1.0 - torch.clamp(zeta, 0.0, 1.0)
    b = 1.0 - torch.clamp(h, min=0.0)
    return a + b - torch.sqrt(a * a + b * b + 1e-24)


def _make_block_and_calibration():
    calibration = {
        "DiscFac": 0.9,
        "CRRA": 2.0,
        "rbar": 1.04,
        "rho_r": 0.2,
        "rho_p": 0.999,
        "rho_q": 0.9,
        "rho_delta": 0.2,
        "sigma_r": 0.001,
        "sigma_p": 0.0001,
        "sigma_q": 0.001,
        "sigma_delta": 0.001,
        "wmin": 0.1,
        "wmax": 4.0,
    }

    from skagent.distributions import Normal

    block = DBlock(
        **{
            "name": "consumption_savings_fb_normalized",
            "description": "Normalized consumption-savings with AR(1) shocks and borrowing constraint via zeta in [0,1] and nonnegative h",
            "shocks": {
                "eps_r": (Normal, {"mu": 0.0, "sigma": "sigma_r"}),
                "eps_delta": (Normal, {"mu": 0.0, "sigma": "sigma_delta"}),
                "eps_p": (Normal, {"mu": 0.0, "sigma": "sigma_p"}),
                "eps_q": (Normal, {"mu": 0.0, "sigma": "sigma_q"}),
            },
            "dynamics": {
                "zeta": Control(
                    ["r", "delta", "q", "p", "w"],
                    lower_bound=lambda r, delta, q, p, w: 0.0,
                    upper_bound=lambda r, delta, q, p, w: 1.0,
                    agent="consumer",
                ),
                "h": Control(
                    ["r", "delta", "q", "p", "w"],
                    lower_bound=lambda r, delta, q, p, w: 0.0,
                    agent="consumer",
                ),
                "c": lambda zeta, w: zeta * w,
                "r": lambda rho_r, r, eps_r: rho_r * r + eps_r,
                "delta": lambda rho_delta, delta, eps_delta: rho_delta * delta
                + eps_delta,
                "p": lambda rho_p, p, eps_p: rho_p * p + eps_p,
                "q": lambda rho_q, q, eps_q: rho_q * q + eps_q,
                "w": lambda w, c, r, p, q, rbar: (
                    (w - c) * rbar * torch.exp(torch.as_tensor(r, dtype=torch.float32))
                    + torch.exp(torch.as_tensor(p, dtype=torch.float32))
                    * torch.exp(torch.as_tensor(q, dtype=torch.float32))
                ),
                "u": lambda c, CRRA: (
                    torch.log(
                        torch.clamp(torch.as_tensor(c, dtype=torch.float32), min=1e-12)
                    )
                    if CRRA == 1
                    else torch.as_tensor(c, dtype=torch.float32) ** (1 - CRRA)
                    / (1 - CRRA)
                ),
            },
            "reward": {"u": "consumer"},
            # Provide notebook-accurate custom residuals (proper functions for readability)
            "resid": {
                "euler": euler_residual,
                "fb": fb_residual,
            },
        }
    )

    return block, calibration


def test_mmw_euler_fb_training_converges_and_small_loss():
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    block, calibration = _make_block_and_calibration()

    givens = prepare_aio_training_inputs(
        block,
        calibration,
        n=256,
        seed=TEST_SEED,
        shock_copies=2,
        state_variables=["r", "delta", "q", "p", "w"],
    )

    policy_net = BlockPolicyNet(block, width=32, n_layers=3)

    loss_fn = get_euler_fb_loss(
        state_variables=["r", "delta", "q", "p", "w"],
        block=block,
        parameters=calibration,
    )

    initial_loss = aggregate_net_loss(
        givens, policy_net.get_decision_function(), loss_fn
    ).detach()

    policy_net = train_block_policy_nn(policy_net, givens, loss_fn, epochs=1500)

    final_loss = aggregate_net_loss(
        givens, policy_net.get_decision_function(), loss_fn
    ).detach()

    assert final_loss.item() < initial_loss.item()
    # Allow a small tolerance consistent with stochastic training and MC error
    assert final_loss.item() < 2e-4
