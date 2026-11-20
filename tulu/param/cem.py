"""方法 A：交叉熵方法（CEM）调参，搜索参数化策略的最优 θ。"""
import json
import time
from pathlib import Path

import numpy as np

from tulu.env.core import *
from tulu.param.policy import *
from tulu.common.stats import summarize

LOWER = np.zeros(THETA_LEN)
LOWER[T_B:T_B + 7] = -50.0               # 各法则的额外价值可为负
UPPER = np.full(THETA_LEN, 1e4)
UPPER[T_B:T_B + 7] = 100.0


def initial_sigma():
    sig = np.zeros(THETA_LEN)
    sig[T_MU] = 2.0
    sig[T_PZ] = 6.0
    sig[T_B:T_B + 7] = 8.0
    sig[T_PHI:T_PHI + 6] = 80.0
    return sig


def cem(cfg, theta0=None, iters=30, pop=64, elite=10, G=3000, K=12, seed=0,
        seed_base=10_000_000, verbose=True, log=print):
    """返回 (最后一轮精英均值, 历史最佳候选)。同一轮内所有候选共用随机种子（CRN），降低比较噪声。"""
    rng = np.random.default_rng(seed)
    mean = default_theta() if theta0 is None else np.array(theta0, dtype=float).copy()
    sig = initial_sigma()
    best_theta, best_score = mean.copy(), -1e18
    for it in range(iters):
        seed0 = seed_base + it * 100_000
        cands = mean + sig * rng.standard_normal((pop, THETA_LEN))
        cands[0] = mean
        cands = np.clip(cands, LOWER, UPPER)
        scores = np.zeros(pop)
        for p in range(pop):
            scores[p] = eval_policy(cands[p], cfg, G, seed0, K, 0)[:, ST_TOTAL].mean()
        idx = np.argsort(-scores)[:elite]
        el = cands[idx]
        mean = el.mean(axis=0)
        sig = el.std(axis=0) * 0.9 + 1e-3 * np.maximum(1.0, np.abs(mean))
        if scores[idx[0]] > best_score:
            best_score, best_theta = scores[idx[0]], cands[idx[0]].copy()
        if verbose:
            log(f"  iter {it:3d} best={scores[idx[0]]:8.2f} elite-mean={scores[idx].mean():8.2f} "
                f"mean-cand={scores[0]:8.2f}")
    return mean, best_theta


def evaluate(theta, cfg, n=100000, seed0=500_000_000, K=12, mode=0):
    """返回 (n, ST_LEN) 的每局统计矩阵。mode=1 为随机合法动作基线。"""
    return eval_policy(np.asarray(theta, dtype=float), cfg, n, seed0, K, mode)


def pick_better(cands, cfg, n=100000, seed0=300_000_000):
    outs = [evaluate(c, cfg, n=n, seed0=seed0) for c in cands]
    return cands[int(np.argmax([o[:, ST_TOTAL].mean() for o in outs]))]


def save_theta(path, theta, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"theta": list(map(float, theta)), "meta": meta or {}},
                                     ensure_ascii=False, indent=2), encoding="utf-8")


def load_theta(path):
    return np.array(json.loads(Path(path).read_text(encoding="utf-8"))["theta"], dtype=float)


def describe_theta(theta):
    names = ["易象", "增辉", "敛芒", "昭示", "引辰", "拓途", "重衍"]
    lines = [f"铭文影子价格 μ = {theta[T_MU]:.2f}",
             f"灾星稀缺惩罚 pZ = {theta[T_PZ]:.2f}",
             "各法则额外价值 b: " + ", ".join(f"{n}={theta[T_B + i]:.1f}" for i, n in enumerate(names)),
             "各回合触腕伤害未来价值 φ: " + ", ".join(f"{theta[T_PHI + i]:.0f}" for i in range(6))]
    return "\n".join(lines)
