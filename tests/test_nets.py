"""方法 B 的快速检查（需要 PyTorch，未安装则跳过）。只做形状、掩码、GAE 数值与一次极小的更新，不是训练。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tulu.config import load_cfg
from tulu.rl.networks import ActorCritic
from tulu.rl.ppo import PPOConfig, compute_gae, collect, update
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS


def test_masked_policy_never_picks_illegal():
    net = ActorCritic()
    obs = torch.rand(16, OBS_DIM)
    mask = torch.zeros(16, NUM_ACTIONS, dtype=torch.bool)
    mask[:, [3, 17, 30]] = True
    for _ in range(5):
        a, logp, v = net.act(obs, mask)
        assert set(a.tolist()) <= {3, 17, 30}
    logits, _ = net(obs, mask)
    p = torch.softmax(logits, -1)
    assert p[:, ~mask[0]].max().item() < 1e-6


def test_gae_tiny_example():
    rew = np.array([[1.0], [1.0]])
    val = np.zeros((2, 1))
    done = np.array([[0.0], [1.0]])
    adv, ret = compute_gae(rew, val, done, np.array([5.0]), 1.0, 1.0)
    assert np.allclose(adv[:, 0], [2.0, 1.0])
    assert np.allclose(ret[:, 0], [2.0, 1.0])


def test_one_tiny_update_runs():
    cfg = PPOConfig(n_envs=8, rollout_len=8, epochs=1, minibatch=32)
    env = VecEnv(load_cfg(), cfg.n_envs, seed=0, parallel=False)
    obs, mask = env.reset()
    net = ActorCritic(hidden=32, layers=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch, obs, mask, totals = collect(env, net, obs, mask, cfg.rollout_len, torch.device("cpu"), cfg.rew_scale)
    st = update(net, opt, batch, cfg, 0.01, torch.device("cpu"))
    assert all(np.isfinite(v) for v in st.values())
