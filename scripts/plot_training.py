"""把训练曲线画成图（results/rl/<name>/training_curve.png），并画上方法 A 与基线的水平参考线。
训练过程中可以随时运行，PyCharm 里直接打开生成的 PNG 查看。
用法：python scripts/plot_training.py --name run1"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tulu.config import RESULTS_DIR, final_nuke

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1")
    ap.add_argument("--param", default="final", help="方法 A 结果目录名，用来画参考线")
    a = ap.parse_args()
    nuke = final_nuke()
    rows = list(csv.DictReader(open(RESULTS_DIR / "rl" / a.name / "train_log.csv", encoding="utf-8")))
    it = [int(r["iter"]) + 1 for r in rows]
    ret = [float(r["mean_return"]) + nuke for r in rows]
    ent = [float(r["entropy"]) for r in rows]
    ev = [(int(r["iter"]) + 1, float(r["eval_mean"]) + nuke) for r in rows if r["eval_mean"] not in ("", None)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(it, ret, lw=1.2, label="训练回报（滑动平均，采样策略）")
    if ev:
        ax[0].plot(*zip(*ev), "o-", color="tab:red", label="贪心评估（2 万局）")
    summ = RESULTS_DIR / "param" / a.param / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text(encoding="utf-8"))
        for key, color in [("最优(参数化)", "tab:green"), ("贪心+法则", "tab:orange"),
                           ("只放牌不打法则", "tab:gray"), ("随机合法动作", "tab:purple")]:
            if key in s:
                ax[0].axhline(s[key]["mean"], ls="--", color=color, lw=1, label=f"{key} {s[key]['mean']:.0f}")
    ax[0].set_xlabel("PPO 迭代"); ax[0].set_ylabel("总倍率（含忘却前夜 +80）")
    ax[0].set_title("方法 B 训练曲线 vs 方法 A 与基线"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(it, ent, color="tab:blue")
    ax[1].set_xlabel("PPO 迭代"); ax[1].set_ylabel("策略熵"); ax[1].set_title("策略熵（越低越确定）"); ax[1].grid(alpha=.3)
    plt.tight_layout()
    out = RESULTS_DIR / "rl" / a.name / "training_curve.png"
    plt.savefig(out, dpi=130)
    print("saved", out, f"(最新迭代 {it[-1]}，回报 {ret[-1]:.1f})")


if __name__ == "__main__":
    main()
