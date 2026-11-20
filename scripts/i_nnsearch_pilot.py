"""[专家迭代] 以网络为基础策略的滚动前瞻试点：配对比较「纯贪心网络」与「网络 + 搜索」。
示例: python scripts/i_nnsearch_pilot.py --games 200 --k 4 --R 32"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulu.config import load_cfg, final_nuke
from tulu.rl.ppo import load_net
from tulu.rl.nnsearch import play_games

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="models/ppo_ft3_last_1296.pt")
ap.add_argument("--games", type=int, default=200)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--R", type=int, default=32)
ap.add_argument("--margin", type=float, default=0.0)
ap.add_argument("--seed0", type=int, default=7_000_000)
ap.add_argument("--base-games", type=int, default=0, help="额外只跑纯贪心的局数（精确基线）")
a = ap.parse_args()
cfg = load_cfg(None); nuke = final_nuke(None)
net, _ = load_net(a.net, "cpu")
t0 = time.time()
base, _ = play_games(net, cfg, a.games, a.seed0, search=False)
print(f"纯贪心 {a.games} 局: {base.mean() + nuke:.2f} ± {base.std() / np.sqrt(a.games):.2f}", flush=True)
srch, frac = play_games(net, cfg, a.games, a.seed0, search=True, k=a.k, R=a.R, margin=a.margin,
                        log=lambda s: None)
d = srch - base
print(f"网络+搜索 k={a.k} R={a.R} margin={a.margin}: {srch.mean() + nuke:.2f} ± {srch.std() / np.sqrt(a.games):.2f}；"
      f"配对差 {d.mean():+.2f} ± {d.std() / np.sqrt(a.games):.2f}；改选比例 {frac:.1%}；用时 {time.time() - t0:.0f}s", flush=True)
