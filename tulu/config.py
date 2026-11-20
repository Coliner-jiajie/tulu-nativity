"""配置加载：从 configs/default.json 读取面板与规则参数，推导 ρ = DEF/ATK 与 h = H/ATK，生成环境配置向量。"""
import json
from pathlib import Path

import numpy as np

from tulu.env.core import make_cfg

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def derive_from_panel(p):
    """由面板推导 ATK、DEF、H、ρ、h。造物加成为加法叠加；黄金造物属于「其他造物」，每个视为 2 个白银。"""
    n_eq = p["silver"] + p["other_counts_as_silver"] * p["gold"]
    boost = 1.0 + p["stat_bonus"]
    atk = p["atk"] * boost * (1 + p["silver_atk"] * n_eq)
    de = p["def"] * boost * (1 + p["silver_def"] * n_eq)
    hp_base = p["con"] * boost * p["hp_per_100_con"] / 100.0
    hp = hp_base * (1 + p["tide_hp_bonus"] * p["tide_layers_lost"])
    return {"n_eq": n_eq, "ATK": atk, "DEF": de, "HP_base": hp_base, "H": hp,
            "rho": de / atk, "h": hp / atk}


def load_params(path=None):
    path = Path(path) if path else CONFIG_DIR / "default.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_cfg(path=None, **override):
    """返回环境配置向量。override 的键同 tulu.env.core.make_cfg（如 rho、h、I0、after、cap）。"""
    params = load_params(path)
    d = derive_from_panel(params["panel"])
    kw = dict(params["rules"])
    kw["rho"] = d["rho"]
    kw["h"] = d["h"]
    kw.update(override)
    return make_cfg(**kw)


def final_nuke(path=None):
    return float(load_params(path).get("final_nuke", 80.0))


if __name__ == "__main__":
    d = derive_from_panel(load_params()["panel"])
    for k, v in d.items():
        print(f"{k:8s} {v:.4f}")
