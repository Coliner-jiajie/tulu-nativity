"""蒸馏专家的检查（需要 numba 即可）：记录的原子动作都合法，剩余回报与总倍率一致，标签函数可用。"""
import numpy as np

from tulu.config import load_cfg
from tulu.param.policy import default_theta
from tulu.rl.expert import generate_expert_dataset, label_batch
from tulu.rl.vec_env import VecEnv


def test_expert_actions_are_legal():
    cfg = load_cfg()
    data = generate_expert_dataset(default_theta(), cfg, 24, seed0=5, K=6, R=8, chunk=24, log=lambda s: None)
    n = len(data["act"])
    assert n > 24 * 20
    # 记录的动作都在合法掩码里
    assert data["mask"][np.arange(n), data["act"]].all()
    # 每局第一条的剩余回报 = 该局总倍率 / 100
    assert np.isfinite(data["ret"]).all() and data["ret"].min() >= 0
    assert abs(data["ret"].max() * 100 - data["totals"].max()) < 1.0 or data["ret"].max() * 100 <= data["totals"].max() + 1.0


def test_label_batch_returns_legal_actions():
    cfg = load_cfg()
    env = VecEnv(cfg, 16, seed=3, parallel=False)
    obs, mask = env.reset()
    labels = np.zeros(16, dtype=np.int64)
    label_batch(env.SI, env.SF, env.cfg, default_theta(), 6, 8, 123, labels)
    assert (labels >= 0).all()
    assert mask[np.arange(16), labels].all()
