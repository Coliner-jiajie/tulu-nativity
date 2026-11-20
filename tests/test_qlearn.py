"""Q 路线的快速检查（需要 PyTorch）：Q 专家数据合法、离线训练能跑、决斗式网络与 evaluate/load 接口兼容、DQN 微调几步不报错。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tulu.config import load_cfg
from tulu.param.policy import default_theta
from tulu.rl.networks import DuelingQNet
from tulu.rl.qexpert import generate_q_dataset
from tulu.rl.qlearn import train_q_offline, dqn_finetune
from tulu.rl.vec_env import NUM_ACTIONS


def _small_data():
    cfg = load_cfg()
    return cfg, generate_q_dataset(default_theta(), cfg, 24, seed0=11, K=6, R=8, chunk=24, log=lambda s: None)


def test_q_dataset_is_consistent():
    cfg, d = _small_data()
    n = len(d["act"])
    assert d["qv"].shape == (n, NUM_ACTIONS)
    # 每个状态都有被专家选中的动作的 Q 值，且该动作合法
    assert d["mask"][np.arange(n), d["act"]].all()
    assert (~np.isnan(d["qv"][np.arange(n), d["act"]])).all()
    # 有 Q 值的动作都是合法动作
    has = ~np.isnan(d["qv"])
    assert (d["mask"][has]).all()
    # qv 是剩余回报：与实际剩余回报同口径（最优候选的价值应与 ret×100 正相关）
    best = np.nanmax(np.where(has, d["qv"], -1e18), axis=1)
    assert np.corrcoef(best, d["ret"] * 100)[0, 1] > 0.5
    assert best.min() > -50


def test_offline_train_and_interfaces(tmp_path):
    cfg, d = _small_data()
    net = DuelingQNet(hidden=32, layers=2)
    train_q_offline(net, d, torch.device("cpu"), epochs=2, batch=256, log=lambda s: None)
    o = torch.from_numpy(d["obs"][:8].astype(np.float32))
    m = torch.from_numpy(d["mask"][:8])
    a, _, _ = net.act(o, m)
    assert m[torch.arange(8), a].all()          # 贪心动作必合法
    from tulu.rl.ppo import save_checkpoint, load_net, PPOConfig
    p = tmp_path / "q.pt"
    save_checkpoint(p, net, torch.optim.Adam(net.parameters()), 0, PPOConfig(), cfg, {})
    net2, _ = load_net(p)
    assert isinstance(net2, DuelingQNet)


def test_dqn_finetune_runs():
    cfg, d = _small_data()
    net = DuelingQNet(hidden=32, layers=1)
    dqn_finetune(net, cfg, torch.device("cpu"), steps=6, n_envs=16, nstep=3, batch=32, buffer_size=4000,
                 updates_per_step=1, demo=d, log=lambda s: None)
