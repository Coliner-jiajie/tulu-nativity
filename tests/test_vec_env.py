"""向量化环境测试（不需要 PyTorch）：观测/掩码形状、掩码合法性、奖励求和等于每局总倍率、场景构造。"""
import numpy as np

from tulu.config import load_cfg, derive_from_panel, load_params
from tulu.env.core import ST_TOTAL, C_RHO, C_H
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS, A_LAW0, A_FREE0, A_OPT0
from tulu.rl.scenario import make_state, observe
from tulu.common.names import parse_card, parse_law, action_names


def test_config_derivation():
    d = derive_from_panel(load_params()["panel"])
    assert abs(d["ATK"] - 337 * 4.6) < 1e-9          # 初始攻击 337（已含 +30%）× 造物
    assert abs(d["scale"] - 337 / (194 * 1.3)) < 1e-12
    assert abs(d["rho"] - 0.20816) < 1e-4
    assert abs(d["h"] - 3.137) < 5e-3
    cfg = load_cfg()
    assert abs(cfg[C_RHO] - d["rho"]) < 1e-12 and abs(cfg[C_H] - d["h"]) < 1e-12


def test_random_masked_play_is_consistent():
    env = VecEnv(load_cfg(), 32, seed=1, parallel=False)
    obs, mask = env.reset()
    assert obs.shape == (32, OBS_DIM) and mask.shape == (32, NUM_ACTIONS)
    rng = np.random.default_rng(1)
    acc = np.zeros(32)
    episodes = 0
    for _ in range(400):
        assert mask.any(axis=1).all()
        a = np.array([rng.choice(np.nonzero(m)[0]) for m in mask])
        obs, mask, rew, done, fin = env.step(a)
        acc += rew
        for e in np.nonzero(done)[0]:
            assert abs(acc[e] - env.fin[e, ST_TOTAL]) < 1e-6
            acc[e] = 0.0
            episodes += 1
        assert np.isfinite(obs).all()
    assert env.bad.sum() == 0
    assert episodes > 0


def test_scenario_obs_and_mask():
    cfg = load_cfg()
    si, sf = make_state(cfg, hand=["灾2", "灾2", "垣1", "蕴5"], laws=["重衍", ("引辰", 0)], I=50, t=1)
    obs, mask = observe(si, sf, cfg)
    assert abs(obs[parse_card("灾2")] - 0.5) < 1e-6          # 两张 /4
    assert mask[parse_card("灾2")] and not mask[parse_card("灾3")]
    assert mask[A_LAW0 + parse_law("重衍")]                    # 付费重衍可打
    assert mask[A_FREE0 + parse_law("引辰")]                   # 0 消耗引辰可打
    assert not mask[A_LAW0 + parse_law("引辰")]
    # 铭文不足时付费法则不可打
    si, sf = make_state(cfg, hand=["灾2", "灾3", "垣1", "蕴5"], laws=["重衍"], I=4, t=1)
    obs, mask = observe(si, sf, cfg)
    assert not mask[A_LAW0 + parse_law("重衍")]
    assert len(action_names()) == NUM_ACTIONS
