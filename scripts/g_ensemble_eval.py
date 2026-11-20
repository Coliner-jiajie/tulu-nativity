"""[评估] 把多个检查点集成（动作概率平均）后评估，并分别评估各单个网络作对照。
用法：python scripts/g_ensemble_eval.py --ckpts models/ppo_ft1_best_1290.pt models/ppo_ft2_best_1293.pt --games 100000"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke
from tulu.rl.ensemble import EnsembleNet
from tulu.rl.evaluate import evaluate_net
from tulu.rl.networks import pick_device
from tulu.rl.ppo import load_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--singles", action="store_true", help="同时评估每个单独网络")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    dev = pick_device(a.device)
    nets = [load_net(p, dev)[0] for p in a.ckpts]
    if a.singles:
        for p, n in zip(a.ckpts, nets):
            r = evaluate_net(n, cfg, n_games=a.games, seed=987654, device=dev)
            print(f"单个 {Path(p).name:34s} {r['mean'] + nuke:8.2f} ± {r['se']:.2f}", flush=True)
    r = evaluate_net(EnsembleNet(nets), cfg, n_games=a.games, seed=987654, device=dev)
    print(f"集成({len(nets)} 个)                                {r['mean'] + nuke:8.2f} ± {r['se']:.2f}  "
          f"sd={r['sd']:.1f}  p5/50/95={r['p05'] + nuke:.0f}/{r['p50'] + nuke:.0f}/{r['p95'] + nuke:.0f}")


if __name__ == "__main__":
    main()
