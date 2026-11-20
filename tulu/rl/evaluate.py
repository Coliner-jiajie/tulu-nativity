"""方法 B：评估神经网络策略（需要 PyTorch）。每个并行环境只统计各自前 k 局，避免偏向短局的采样偏差。"""
import numpy as np
import torch

from tulu.env.core import ST_TOTAL, ST_LEN
from tulu.rl.vec_env import VecEnv


def evaluate_net(net, env_cfg, n_games=20000, seed=123, n_envs=2000, greedy=True, device="cpu",
                 return_matrix=False):
    n_envs = min(n_envs, n_games)
    k = int(np.ceil(n_games / n_envs))
    env = VecEnv(env_cfg, n_envs, seed=seed, parallel=True)
    obs, mask = env.reset()
    net.eval()
    counts = np.zeros(n_envs, dtype=np.int64)
    rows = []
    guard = 0
    while counts.min() < k and guard < 100000:
        guard += 1
        o = torch.from_numpy(obs).to(device)
        m = torch.from_numpy(mask).to(device)
        a, _, _ = net.act(o, m, greedy=greedy)
        obs, mask, _, done, _ = env.step(a.cpu().numpy())
        idx = np.nonzero(done)[0]
        for e in idx:
            if counts[e] < k:
                rows.append(env.fin[e].copy())
                counts[e] += 1
    mat = np.array(rows)[:n_games]
    r = mat[:, ST_TOTAL]
    res = {"mean": float(r.mean()), "se": float(r.std() / np.sqrt(len(r))), "sd": float(r.std()),
           "p05": float(np.percentile(r, 5)), "p50": float(np.percentile(r, 50)),
           "p95": float(np.percentile(r, 95)), "n": int(len(r))}
    return (res, mat) if return_matrix else res
