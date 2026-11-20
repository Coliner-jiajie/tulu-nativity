"""构造指定局面（手牌、法则、星轨、铭文、回合、触腕伤害），用于检查训练好的策略在典型局面下怎么打。"""
import numpy as np

from tulu.env.core import *
from tulu.common.names import parse_card, parse_law
from tulu.rl.vec_env import build_obs_mask, NUM_ACTIONS, OBS_DIM


def make_state(cfg, hand, laws=(), track=(), I=50, t=1, B=None):
    """hand/track: ['灾2', '灾3', ...]；laws: ['重衍', ('引辰', 0), ...]，字符串默认消耗 5，元组为 (名称, 消耗)。
    抽牌堆 = 整副牌 45 张扣除手牌与星轨中的牌。"""
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    for c in range(NT):
        si[O_DP + c] = 3
    for s in hand:
        c = parse_card(s)
        si[O_DP + c] -= 1
        si[O_HO + si[O_HN]] = c
        si[O_HC + si[O_HN]] = c
        si[O_HN] += 1
    for s in track:
        c = parse_card(s)
        si[O_DP + c] -= 1
        si[O_TO + si[O_TN]] = c
        si[O_TC + si[O_TN]] = c
        si[O_TN] += 1
    for lw in laws:
        name, cost = (lw, int(cfg[C_LAWCOST])) if isinstance(lw, str) else lw
        si[O_LI + si[O_LN]] = parse_law(name)
        si[O_LC + si[O_LN]] = cost
        si[O_LN] += 1
    si[O_I] = I
    si[O_T] = t
    sf[F_B] = cfg[C_B0] * cfg[C_H] + max(0, t - 1) * cfg[C_GROW] * cfg[C_H] * cfg[C_M] if B is None else B
    return si, sf


def observe(si, sf, cfg):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    build_obs_mask(si, sf, np.asarray(cfg, dtype=np.float64), obs, mask)
    return obs, mask
