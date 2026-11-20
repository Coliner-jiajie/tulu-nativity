"""方法 A：参数扫描。对某个配置参数（如 rho、h、I0、after、cap）取一组值，
每个值用 CEM 求参数化最优策略，再大样本评估，并与基线对比。"""
import json
import time
from pathlib import Path

import numpy as np

from tulu.config import load_cfg, final_nuke
from tulu.env.core import ST_TOTAL
from tulu.param.cem import cem, evaluate, pick_better
from tulu.param.policy import no_law_theta, greedy_theta, default_theta
from tulu.common.stats import summarize, fmt_line


def run_config(tag, cfg, warm=None, iters=25, pop=64, G=3000, nfinal=200000, nuke=0.0, log=print):
    t0 = time.time()
    mean, best = cem(cfg, theta0=warm, iters=iters, pop=pop, G=G, verbose=False)
    theta = pick_better([mean, best], cfg)
    s = summarize(evaluate(theta, cfg, n=nfinal), nuke)
    s["theta"] = list(map(float, theta))
    s["random"] = summarize(evaluate(default_theta(), cfg, n=100000, mode=1), nuke)["mean"]
    s["nolaw"] = summarize(evaluate(no_law_theta(), cfg, n=100000), nuke)["mean"]
    s["greedy"] = summarize(evaluate(greedy_theta(), cfg, n=100000), nuke)["mean"]
    s["time"] = time.time() - t0
    log(f"[{tag}] " + fmt_line("最优(参数化)", s) +
        f" | random={s['random']:.0f} nolaw={s['nolaw']:.0f} greedy={s['greedy']:.0f} ({s['time']:.0f}s)")
    return np.array(theta), s


def run_sweep(param, values, out_path, config_path=None, iters=25, pop=64, G=3000, nfinal=200000,
              warm_start=True, log=print):
    """param: make_cfg 的参数名（rho、h、I0、after、cap、...）。结果追加写入 out_path（JSON）。"""
    nuke = final_nuke(config_path)
    results = []
    warm = None
    for v in values:
        cfg = load_cfg(config_path, **{param: v})
        theta, s = run_config(f"{param}={v}", cfg, warm if warm_start else None,
                              iters=iters, pop=pop, G=G, nfinal=nfinal, nuke=nuke, log=log)
        warm = theta
        results.append({param: v, **s})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results
