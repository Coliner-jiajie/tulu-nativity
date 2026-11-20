"""统计汇总：把 eval_policy / 向量化评估得到的每局统计矩阵汇总成字典。"""
import numpy as np

from tulu.env.core import (ST_TOTAL, ST_NLAW, ST_LAWCNT, ST_TURNR, ST_N3Z, ST_N3Y, ST_N3H,
                           ST_BFIN, ST_ILEFT, ST_TURNY, ST_TURNZ, ST_REFUND, ST_CYBANK, ST_LOCKLAW)


def summarize(out, final_nuke=0.0):
    """out: (局数, ST_LEN)。final_nuke 为退场「忘却前夜」的常数（80.0），加到总倍率上。"""
    r = out[:, ST_TOTAL] + final_nuke
    n = len(r)
    return {
        "n": n,
        "mean": float(r.mean()),
        "se": float(r.std() / np.sqrt(n)),
        "sd": float(r.std()),
        "p05": float(np.percentile(r, 5)),
        "p50": float(np.percentile(r, 50)),
        "p95": float(np.percentile(r, 95)),
        "laws": float(out[:, ST_NLAW].mean()),
        "lawcnt": out[:, ST_LAWCNT:ST_LAWCNT + 7].mean(axis=0).tolist(),
        "n3Z": float(out[:, ST_N3Z].mean()),
        "n3Y": float(out[:, ST_N3Y].mean()),
        "n3H": float(out[:, ST_N3H].mean()),
        "Bfin": float(out[:, ST_BFIN].mean()),
        "Ileft": float(out[:, ST_ILEFT].mean()),
        "refund": float(out[:, ST_REFUND].mean()),
        "cybank": float(out[:, ST_CYBANK].mean()),
        "locklaw": float(out[:, ST_LOCKLAW].mean()),
        "turnR": out[:, ST_TURNR:ST_TURNR + 6].mean(axis=0).tolist(),
        "turnY": out[:, ST_TURNY:ST_TURNY + 6].mean(axis=0).tolist(),
        "turnZ": out[:, ST_TURNZ:ST_TURNZ + 6].mean(axis=0).tolist(),
    }


def fmt_line(name, s):
    return (f"{name:24s} mean={s['mean']:8.2f} ± {s['se']:.2f}  sd={s['sd']:6.1f}  "
            f"p5/50/95={s['p05']:.0f}/{s['p50']:.0f}/{s['p95']:.0f}  laws={s['laws']:.2f}  "
            f"3灾={s['n3Z']:.2f}  余波+{s['refund']:.2f}  白嫖重衍={s['cybank']:.2f}")
