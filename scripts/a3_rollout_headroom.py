"""[方法 A·检验] 滚动前瞻：估计参数化策略离最优还有多远（每个决策点用基础策略模拟完整局选动作）。
用法：python scripts/a3_rollout_headroom.py --theta results/param/final/theta.json --games 1000 --R 200"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke
from tulu.env.core import ST_TOTAL
from tulu.param.cem import load_theta
from tulu.param.policy import eval_policy
from tulu.param.rollout import eval_improved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--theta", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--R", type=int, default=200, help="每个候选动作的模拟局数")
    ap.add_argument("--seed", type=int, default=700_000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    theta = load_theta(a.theta)
    t0 = time.time()
    base = eval_policy(theta, cfg, a.games, a.seed, 12, 0)[:, ST_TOTAL]
    imp = eval_improved(theta, cfg, a.games, a.seed, 12, a.R)[:, ST_TOTAL]
    d = imp - base
    n = a.games
    print(f"基础策略   {base.mean() + final_nuke(a.config):.2f} ± {base.std() / np.sqrt(n):.2f}")
    print(f"滚动前瞻   {imp.mean() + final_nuke(a.config):.2f} ± {imp.std() / np.sqrt(n):.2f}")
    print(f"配对差值   {d.mean():+.2f} ± {d.std() / np.sqrt(n):.2f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
