"""输出层换算：训练口径的归一化倍率（×ATK、含忘却前夜 +80）→ 真实伤害。

只用于显示 / 报告，训练、环境、奖励都不 import 本模块，也不读取 configs/display.json。
乘区关系（按计算器的系统逻辑，已验证）：
    A      = 面板攻击 × 1.3 × (1 + 0.15 × 白银等效造物数)          （即 derive_from_panel 的 ATK）
    Z      = 1.5                         专武
    C      = 1 + 0.30 + 0.02 × 狂气回充   以此身承负
    M铭    = 1 + 0.025 × 层数；灵塑适性时 1 + 0.05 × 层数
    真实伤害 = 归一化倍率 × A × Z × C × M铭
    忘却前夜 = 80 × (3 若以此碑诏告)       （训练口径固定 +80）
"""
import json
from functools import lru_cache
from pathlib import Path

from tulu.config import load_params, derive_from_panel, ROOT

DISPLAY_PATH = ROOT / "configs" / "display.json"


@lru_cache(maxsize=None)
def _load(path_str):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def display_params(path=None, config=None):
    """返回换算参数字典：atk、Z、C、Mming、ext、scale、nuke_train、nuke_real 等。"""
    d = _load(str(path or DISPLAY_PATH))
    atk = derive_from_panel(load_params(config)["panel"])["ATK"]
    z = 1.5 if d.get("zhanwu") else 1.0
    c = (1.0 + 0.30 + 0.02 * float(d.get("kuangqi", 0))) if d.get("chengfu") else 1.0
    per = 0.05 if d.get("lingsu") else 0.025
    m = 1.0 + per * float(d.get("ming_layers", 0))
    ext = z * c * m
    nuke_train = float(d.get("train_nuke", 80.0))
    nuke_real = 80.0 * (3.0 if d.get("beizhao") else 1.0)
    return {"atk": atk, "Z": z, "C": c, "Mming": m, "ext": ext, "scale": atk * ext,
            "nuke_train": nuke_train, "nuke_real": nuke_real,
            "kuangqi": float(d.get("kuangqi", 0)), "lingsu": bool(d.get("lingsu")),
            "zhanwu": bool(d.get("zhanwu")), "chengfu": bool(d.get("chengfu")),
            "beizhao": bool(d.get("beizhao")), "ming_layers": float(d.get("ming_layers", 0))}


def total_to_real(total_train, p=None):
    """训练口径总倍率（6 回合 + 80）→ 真实总伤害（6 回合 + 忘却前夜真实值）。对均值和分位数都成立（仿射变换）。"""
    p = p or display_params()
    return (total_train - p["nuke_train"] + p["nuke_real"]) * p["scale"]


def turns_to_real(r_turns, p=None):
    """仅 6 回合部分的倍率（不含忘却前夜）→ 真实伤害。"""
    p = p or display_params()
    return r_turns * p["scale"]


def fmt_dmg(v):
    """真实伤害的可读写法：≥1 亿用「亿」，≥1 万用「万」。"""
    a = abs(v)
    if a >= 1e8:
        return f"{v / 1e8:.3f} 亿"
    if a >= 1e4:
        return f"{v / 1e4:.1f} 万"
    return f"{v:.0f}"


def real_line(label, mean, p05=None, p50=None, p95=None, p=None):
    """c_compare / b3 的补充输出行：mean 与分位数为训练口径总倍率（含 +80）。"""
    p = p or display_params()
    s = f"{'':4s}↳ 真实伤害 {label}: 均值 {fmt_dmg(total_to_real(mean, p))}"
    if p05 is not None:
        s += (f"  p5/50/95={fmt_dmg(total_to_real(p05, p))}/{fmt_dmg(total_to_real(p50, p))}"
              f"/{fmt_dmg(total_to_real(p95, p))}")
    return s


def describe(p=None):
    p = p or display_params()
    return (f"换算系数 {p['scale']:.2f} = ATK {p['atk']:.2f} × 专武 {p['Z']:.2f} × 承负 {p['C']:.3f}"
            f" × 铭文 {p['Mming']:.3f}；忘却前夜 +{p['nuke_real']:g}（训练口径 +{p['nuke_train']:g}）")


if __name__ == "__main__":
    pp = display_params()
    print(describe(pp))
    print("示例：训练口径 1291 →", fmt_dmg(total_to_real(1291, pp)))
