"""[对比] 把方法 A（参数化 + CEM）与方法 B（神经网络）以及基线放在同一个环境配置下对比。
用法：python scripts/c_compare.py --theta results/param/final/theta.json --ckpt results/rl/run1/best.pt"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke
from tulu.param.cem import evaluate as eval_param, load_theta
from tulu.param.policy import no_law_theta, greedy_theta, default_theta
from tulu.common.stats import summarize, fmt_line
from tulu.display import display_params, real_line, describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--theta", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--games", type=int, default=200000)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    n = a.games
    dp = display_params()
    print(describe(dp))
    show = lambda name, s: (print(fmt_line(name, s)),
                            print(real_line(name, s['mean'], s['p05'], s['p50'], s['p95'], dp)))
    show("随机合法动作", summarize(eval_param(default_theta(), cfg, n=n, mode=1), nuke))
    show("只放牌不打法则", summarize(eval_param(no_law_theta(), cfg, n=n), nuke))
    show("贪心+法则", summarize(eval_param(greedy_theta(), cfg, n=n), nuke))
    if a.theta:
        show("方法A 参数化最优", summarize(eval_param(load_theta(a.theta), cfg, n=n), nuke))
    if a.ckpt:
        from tulu.rl.ppo import load_net
        from tulu.rl.networks import pick_device
        from tulu.rl.evaluate import evaluate_net
        dev = pick_device("auto")
        net, ck = load_net(a.ckpt, dev)
        r = evaluate_net(net, cfg, n_games=n, seed=987654, device=dev)
        print(f"{'方法B 神经网络(贪心)':24s} mean={r['mean'] + nuke:8.2f} ± {r['se']:.2f}  sd={r['sd']:6.1f}  "
              f"p5/50/95={r['p05'] + nuke:.0f}/{r['p50'] + nuke:.0f}/{r['p95'] + nuke:.0f}")
        print(real_line("方法B 神经网络(贪心)", r['mean'] + nuke, r['p05'] + nuke, r['p50'] + nuke, r['p95'] + nuke, dp))


if __name__ == "__main__":
    main()
