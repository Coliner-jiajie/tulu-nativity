"""[方法 A] 用 CEM 求参数化策略的最优参数，大样本评估并与基线对比。结果写入 results/param/<name>/。
用法（在项目根目录）：python scripts/a1_tune_cem.py --name final --iters 30 --pop 64 --games 3000"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke, RESULTS_DIR, derive_from_panel, load_params
from tulu.param.cem import cem, evaluate, pick_better, save_theta, describe_theta
from tulu.param.policy import no_law_theta, greedy_theta, default_theta
from tulu.common.stats import summarize, fmt_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="final")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--games", type=int, default=3000, help="每个候选每轮评估的局数")
    ap.add_argument("--final-games", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", default=None, help="热启动用的 theta JSON")
    a = ap.parse_args()

    out = RESULTS_DIR / "param" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    print("面板推导:", {k: round(v, 4) for k, v in derive_from_panel(load_params(a.config)["panel"]).items()})
    warm = None
    if a.warm:
        from tulu.param.cem import load_theta
        warm = load_theta(a.warm)
    t0 = time.time()
    mean, best = cem(cfg, theta0=warm, iters=a.iters, pop=a.pop, G=a.games, seed=a.seed)
    theta = pick_better([mean, best], cfg)
    s = summarize(evaluate(theta, cfg, n=a.final_games), nuke)
    rows = {"最优(参数化)": s,
            "随机合法动作": summarize(evaluate(default_theta(), cfg, n=100000, mode=1), nuke),
            "只放牌不打法则": summarize(evaluate(no_law_theta(), cfg, n=100000), nuke),
            "贪心+法则": summarize(evaluate(greedy_theta(), cfg, n=100000), nuke)}
    for k, v in rows.items():
        print(fmt_line(k, v))
    print(describe_theta(theta))
    save_theta(out / "theta.json", theta, {"iters": a.iters, "pop": a.pop, "games": a.games})
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
