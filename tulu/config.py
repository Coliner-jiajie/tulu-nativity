"""配置加载：从 configs/default.json 读取面板与规则参数，推导 ρ = DEF/ATK 与 h = H/ATK，生成环境配置向量。"""
import json
from pathlib import Path

import numpy as np

from tulu.env.core import make_cfg

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def panel_stats(p, level=None):
    """某等级的面板（已含诞临阶段灵塑 +stat_bonus，不含造物）：每项 = (基础值 + (等级 − 基础等级) × 每级增长) × (1 + stat_bonus)。
    基础值是官方图（60 级）数值；level 缺省取 panel["level"]。"""
    lv = p["level"] if level is None else level
    boost = 1.0 + p["stat_bonus"]
    return {k: (p["base"][k] + (lv - p["base_level"]) * p["growth_per_level"][k]) * boost for k in ("atk", "def", "con")}


def derive_from_panel(p, level=None):
    """由面板推导 ATK、DEF、H、ρ、h。造物加成为加法叠加；黄金造物属于「其他造物」，每个视为 2 个白银。
    面板随等级线性增长（见 panel_stats）；level 缺省取 panel["level"]，可传入其他等级查看不同等级的数值。"""
    n_eq = p["silver"] + p["other_counts_as_silver"] * p["gold"]
    boost = 1.0 + p["stat_bonus"]
    st = panel_stats(p, level)
    lv = p["level"] if level is None else level
    atk = st["atk"] * (1 + p["silver_atk"] * n_eq)
    de = st["def"] * (1 + p["silver_def"] * n_eq)
    hp_base = st["con"] * p["hp_per_100_con"] / 100.0
    hp = hp_base * (1 + p["tide_hp_bonus"] * p["tide_layers_lost"])
    return {"level": lv, "n_eq": n_eq, "scale": st["atk"] / (p["base"]["atk"] * boost),
            "atk_panel": st["atk"], "def_panel": st["def"], "con_panel": st["con"],
            "ATK": atk, "DEF": de, "HP_base": hp_base, "H": hp, "rho": de / atk, "h": hp / atk}


def load_params(path=None):
    path = Path(path) if path else CONFIG_DIR / "default.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_cfg(path=None, level=None, **override):
    """返回环境配置向量。override 的键同 tulu.env.core.make_cfg（如 rho、h、I0、after、cap）。
    level：覆盖面板等级（目前攻防体质同幅增长，ρ、h 与等级无关）。"""
    params = load_params(path)
    d = derive_from_panel(params["panel"], level)
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
