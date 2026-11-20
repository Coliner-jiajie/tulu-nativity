"""[专家迭代] 用「网络+搜索」打局，记录每个有选择的状态的搜索后动作（改选的才与网络贪心不同）。
结果: results/rl/nnsearch/data_<name>_<i>.npz (obs, mask, act, greedy, total)。"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulu.config import load_cfg, final_nuke
from tulu.rl.ppo import load_net
from tulu.rl.nnsearch import play_games

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="models/ppo_ft3_last_1296.pt")
ap.add_argument("--name", default="s1")
ap.add_argument("--chunks", type=int, default=8)
ap.add_argument("--games", type=int, default=500)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--R", type=int, default=512)
ap.add_argument("--margin", type=float, default=4.0)
ap.add_argument("--seed0", type=int, default=20_000_000)
a = ap.parse_args()
cfg = load_cfg(None); nuke = final_nuke(None)
net, _ = load_net(a.net, "cpu")
out = Path("results/rl/nnsearch"); out.mkdir(parents=True, exist_ok=True)
for i in range(a.chunks):
    rec = {"obs": [], "mask": [], "act": [], "greedy": []}
    t0 = time.time()
    tot, frac = play_games(net, cfg, a.games, a.seed0 + i * a.games, search=True, k=a.k, R=a.R,
                           margin=a.margin, log=lambda s: None, record=rec)
    d = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez_compressed(out / f"data_{a.name}_{i}.npz", total=tot, **d)
    print(f"[{a.name} {i + 1}/{a.chunks}] 搜索局均 {tot.mean() + nuke:.1f}；样本 {len(d['act'])}；改选 "
          f"{int((d['act'] != d['greedy']).sum())}；{time.time() - t0:.0f}s", flush=True)
