"""[可视化] 让训练得到的策略各打若干局，录制成逐帧回放数据，写到 viz/data.js，供 viz/index.html 播放。
用法：python scripts/v_export_episodes.py --games 24 --policies expert,param,ppo
同一个种子 = 同一个开局（相同的初始手牌与法则），方便对比不同策略怎么打同一手牌。"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, load_params, derive_from_panel, final_nuke, ROOT
from tulu.env.core import *
from tulu.param.cem import load_theta
from tulu.viz.recorder import Game, play_param, play_expert, play_ppo
from tulu.display import display_params, total_to_real


def real_meta():
    """输出层换算信息：写进 meta，供 viz/index.html 显示真实伤害。"""
    p = display_params()
    return {"scale": round(p["scale"], 4), "nukeReal": p["nuke_real"], "atk": round(p["atk"], 3),
            "Z": p["Z"], "C": round(p["C"], 4), "Mming": round(p["Mming"], 4),
            "kuangqi": p["kuangqi"], "lingsu": p["lingsu"], "beizhao": p["beizhao"]}


def patch_meta(path):
    """不重跑对局，只把换算信息写进已有的 data.js。"""
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    head = "window.TULU_DATA = "
    data = json.loads(txt[len(head):].rstrip().rstrip(";"))
    data["meta"]["real"] = real_meta()
    path.write_text(head + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("已更新换算信息：", data["meta"]["real"])

POLICIES = {
    "expert": {"name": "滚动前瞻专家", "short": "专家",
               "desc": "目前最强。每个决策点对每个候选动作各模拟 200 局（后续按方法 A 打），选平均总倍率最高的。约 1291。",
               "mean": 1291.0},
    "param": {"name": "方法 A · 参数化最优", "short": "参数化",
              "desc": "15 个参数的打分公式 + 一步前瞻，用 CEM 搜索最优参数；30 万局评估 1229.9。",
              "mean": 1229.9},
    "ppo": {"name": "方法 B · PPO 神经网络", "short": "PPO",
            "desc": "纯强化学习的演员-评论家网络（贪心动作），2000 轮训练；约 1196。",
            "mean": 1196.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--policies", default="expert,param,ppo")
    ap.add_argument("--theta", default=str(ROOT / "results/param/final/theta.json"))
    ap.add_argument("--ckpt", default=str(ROOT / "results/rl/run1/best.pt"))
    ap.add_argument("--out", default=str(ROOT / "viz/data.js"))
    ap.add_argument("--meta-only", action="store_true", help="只更新已有 data.js 里的真实伤害换算信息，不重跑对局")
    a = ap.parse_args()
    if a.meta_only:
        patch_meta(a.out)
        return

    cfg = load_cfg()
    nuke = final_nuke()
    params = load_params()
    d = derive_from_panel(params["panel"])
    theta = np.asarray(load_theta(a.theta), dtype=np.float64)
    wanted = [p for p in a.policies.split(",") if p]
    net = None
    if "ppo" in wanted:
        try:
            from tulu.rl.ppo import load_net
            net, ck = load_net(a.ckpt, "cpu")
        except Exception as e:      # 没有 torch 或模型：跳过 PPO
            print("跳过 PPO：", e)
            wanted.remove("ppo")

    episodes, pols = [], []
    for p in wanted:
        t0 = time.time()
        tot = []
        for s in range(a.seed0, a.seed0 + a.games):
            g = Game(cfg, s)
            if p == "param":
                play_param(g, theta)
            elif p == "expert":
                play_expert(g, theta)
            else:
                play_ppo(g, net)
            ep = g.result(p)
            ep["id"] = f"{p}-{s}"
            episodes.append(ep)
            tot.append(ep["total"])
        info = dict(POLICIES[p])
        info["key"] = p
        info["sample_mean"] = round(float(np.mean(tot)) + nuke, 1)
        info["n"] = len(tot)
        pols.append(info)
        print(f"{p:7s} {len(tot)} 局，样本均值 {np.mean(tot) + nuke:.1f}（含 +{nuke:g}），用时 {time.time() - t0:.0f}s")

    meta = {
        "nuke": nuke, "rho": round(float(cfg[C_RHO]), 4), "h": round(float(cfg[C_H]), 3),
        "m": float(cfg[C_M]), "I0": int(cfg[C_I0]), "lawcost": int(cfg[C_LAWCOST]), "cap": int(cfg[C_CAP]),
        "fZ": float(cfg[C_FZ]), "fH": float(cfg[C_FH]), "fY": float(cfg[C_FY]),
        "policies": pols,
        "real": real_meta(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.TULU_DATA = " + json.dumps({"meta": meta, "episodes": episodes},
                                                       ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"写入 {out}（{out.stat().st_size / 1e6:.2f} MB，{len(episodes)} 局）")


if __name__ == "__main__":
    main()
