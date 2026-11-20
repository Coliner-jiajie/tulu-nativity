"""[可视化] 导出前端引擎需要的参数到 viz/params.js：环境配置向量 cfg、方法 A 的 θ、PPO 网络权重、真实伤害换算。
规则或策略变了就重跑一次：python scripts/v_export_params.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke, ROOT
from tulu.param.cem import load_theta
from tulu.display import display_params


def main():
    cfg = load_cfg()
    out = {"cfg": [float(x) for x in cfg], "theta": [float(x) for x in load_theta(ROOT / "results/param/final/theta.json")],
           "nuke": final_nuke()}
    p = display_params()
    out["real"] = {"scale": round(p["scale"], 4), "nukeReal": p["nuke_real"], "atk": round(p["atk"], 3), "Z": p["Z"],
                   "C": round(p["C"], 4), "Mming": round(p["Mming"], 4), "kuangqi": p["kuangqi"],
                   "lingsu": p["lingsu"], "beizhao": p["beizhao"]}
    txt = "window.TULU_PARAMS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n"
    try:
        import torch
        ck = torch.load(ROOT / "results/rl/run1/best.pt", map_location="cpu", weights_only=False)
        sd = ck["model"]
        r = lambda k: [round(float(x), 5) for x in sd[k].numpy().reshape(-1)]
        model = {"hidden": ck["meta"]["hidden"], "W1": r("trunk.0.weight"), "b1": r("trunk.0.bias"),
                 "W2": r("trunk.2.weight"), "b2": r("trunk.2.bias"), "Wp": r("pi.weight"), "bp": r("pi.bias")}
        assert ck["meta"]["layers"] == 2
        txt += "window.TULU_MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n"
    except Exception as e:
        print("未导出 PPO 权重：", e)
    (ROOT / "viz/params.js").write_text(txt, encoding="utf-8")
    print(f"写入 viz/params.js（{len(txt) / 1e3:.0f} KB）")


if __name__ == "__main__":
    main()
