"""[方法 B] 冒烟测试（不是训练）：用随机合法动作跑一小段向量化环境，检查观测、掩码、奖励与每局总倍率一致。
用法：python scripts/b1_smoke_vecenv.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg
from tulu.env.core import ST_TOTAL
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS


def main(n_envs=64, steps=600, seed=0):
    env = VecEnv(load_cfg(), n_envs, seed=seed, parallel=False)
    obs, mask = env.reset()
    assert obs.shape == (n_envs, OBS_DIM) and mask.shape == (n_envs, NUM_ACTIONS)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n_envs)
    episodes = 0
    max_err = 0.0
    for _ in range(steps):
        assert mask.any(axis=1).all(), "存在没有合法动作的局面"
        a = np.array([rng.choice(np.nonzero(m)[0]) for m in mask])
        obs, mask, rew, done, fin = env.step(a)
        acc += rew
        for e in np.nonzero(done)[0]:
            max_err = max(max_err, abs(acc[e] - env.fin[e, ST_TOTAL]))
            acc[e] = 0.0
            episodes += 1
    assert env.bad.sum() == 0, "出现非法动作"
    print(f"OK: {episodes} episodes, max |sum(reward) - total| = {max_err:.2e}")


if __name__ == "__main__":
    main()
