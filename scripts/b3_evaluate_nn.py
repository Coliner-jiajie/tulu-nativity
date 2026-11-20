"""[方法 B] 评估训练好的神经网络策略（贪心动作），可选与方法 A 的最优参数对比。
用法：python scripts/b3_evaluate_nn.py --ckpt results/rl/run1/best.pt --games 100000
      python scripts/b3_evaluate_nn.py --ckpt results/rl/run1/best.pt --theta results/param/final/theta.json"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke
from tulu.display import display_params, real_line, describe
from tulu.rl.ppo import load_net
from tulu.rl.networks import pick_device
from tulu.rl.evaluate import evaluate_net
from tulu.param.cem import evaluate as eval_param, load_theta
from tulu.common.stats import summarize, fmt_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--theta", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    dev = pick_device(a.device)
    net, ck = load_net(a.ckpt, dev)
    r = evaluate_net(net, cfg, n_games=a.games, seed=987654, device=dev)
    print(f"神经网络(贪心)  mean={r['mean'] + nuke:.2f} ± {r['se']:.2f}  sd={r['sd']:.1f}  "
          f"p5/50/95={r['p05'] + nuke:.0f}/{r['p50'] + nuke:.0f}/{r['p95'] + nuke:.0f}  (训练到第 {ck['iter'] + 1} 轮)")
    dp = display_params()
    print(describe(dp))
    print(real_line("神经网络(贪心)", r['mean'] + nuke, r['p05'] + nuke, r['p50'] + nuke, r['p95'] + nuke, dp))
    if a.theta:
        s = summarize(eval_param(load_theta(a.theta), cfg, n=a.games), nuke)
        print(fmt_line("参数化最优(方法A)", s))
        print(real_line("参数化最优(方法A)", s['mean'], s['p05'], s['p50'], s['p95'], dp))


if __name__ == "__main__":
    main()
