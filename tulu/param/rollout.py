"""方法 A（检验用）：滚动前瞻（rollout）策略改进 v3：在每个决策点，对每个候选动作用基础策略模拟打完整局，
选平均总倍率最高的动作。候选动作：直接补满星轨（FINISH）、每种法则（带基础策略建议的锁定）、
以及不锁定直接打法则。"""
import numpy as np
from numba import njit, prange
from tulu.env.core import *
from tulu.param.policy import *

KIND_FINISH, KIND_LAW, KIND_OPTION = 0, 1, 2


@njit(cache=True)
def apply_action(si, sf, cfg, kind, a, l1, l2, p1, p2, p3, st):
    if kind == KIND_LAW:
        if l1 >= 0:
            place_by_value(si, sf, cfg, l1, st)
        if l2 >= 0:
            place_by_value(si, sf, cfg, l2, st)
        play_law(si, sf, cfg, a, st)
    elif kind == KIND_OPTION:
        pick_option(si, cfg, a)
    else:
        si[O_PLAN] = p1
        si[O_PLAN + 1] = p2
        si[O_PLAN + 2] = p3
        si[O_COMMIT] = 1


@njit(cache=True)
def action_value(si, sf, theta, cfg, K, R, seed_base, kind, a, l1, l2, p1, p2, p3):
    si2 = np.zeros(SI_LEN, dtype=np.int64)
    sf2 = np.zeros(SF_LEN)
    st = np.zeros(ST_LEN)
    tot = 0.0
    for r in range(R):
        si2[:] = si
        sf2[:] = sf
        st[:] = 0.0
        np.random.seed(seed_base + r)
        apply_action(si2, sf2, cfg, kind, a, l1, l2, p1, p2, p3, st)
        play_until_done(si2, sf2, theta, cfg, K, 0, st)
        tot += sf2[F_R]
    return tot / R


@njit(cache=True)
def improved_game(theta, cfg, K, R, ntri, gseed, st):
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    np.random.seed(gseed)
    init_game(si, sf, cfg)
    dec = 0
    plan0 = np.zeros(3, dtype=np.int64)
    bplan = np.zeros(3, dtype=np.int64)
    plans = np.zeros((8, 3), dtype=np.int64)
    scs = np.zeros(8)
    out = np.zeros(2, dtype=np.int64)
    while si[O_DONE] == 0 and dec < 600:
        dec += 1
        sb = gseed * 7919 + dec * 104729
        t = int(si[O_T])
        hn = si[O_HN]
        tn = si[O_TN]
        V0 = best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0)
        if si[O_PN] > 0:
            bv = -1e18
            bj = 0
            for j in range(si[O_PN]):
                v = action_value(si, sf, theta, cfg, K, R, sb, KIND_OPTION, j, -1, -1, 0, 0, 0)
                if v > bv:
                    bv = v
                    bj = j
            pick_option(si, cfg, bj)
        elif si[O_COMMIT] == 1:
            place_by_value(si, sf, cfg, si[O_PLAN + si[O_TN]], st)
        else:
            best = -1e18
            bkind = -1
            ba = 0
            bl1 = -1
            bl2 = -1
            if V0 > -1e17:
                nk = top_completions(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta,
                                     min(ntri, 8), plans, scs)
                for q in range(nk):
                    v = action_value(si, sf, theta, cfg, K, R, sb, KIND_FINISH, 0, -1, -1,
                                     plans[q, 0], plans[q, 1], plans[q, 2])
                    if v > best:
                        best = v
                        bkind = KIND_FINISH
                        for z in range(3):
                            bplan[z] = plans[q, z]
            for li in range(si[O_LN]):
                lid = si[O_LI + li]
                cost = si[O_LC + li]
                if si[O_I] < cost:
                    continue
                dup = False
                for lj in range(li):
                    if si[O_LI + lj] == lid and si[O_LC + lj] == cost:
                        dup = True
                if dup:
                    continue
                law_value2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out)
                l1 = out[0]
                l2 = out[1]
                v = action_value(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, l1, l2, 0, 0, 0)
                if v > best:
                    best = v
                    bkind = KIND_LAW
                    ba = li
                    bl1 = l1
                    bl2 = l2
                if l1 >= 0 or l2 >= 0:
                    v = action_value(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, -1, -1, 0, 0, 0)
                    if v > best:
                        best = v
                        bkind = KIND_LAW
                        ba = li
                        bl1 = -1
                        bl2 = -1
            if bkind == KIND_LAW:
                apply_action(si, sf, cfg, KIND_LAW, ba, bl1, bl2, 0, 0, 0, st)
            elif bkind == KIND_FINISH:
                si[O_PLAN] = bplan[0]
                si[O_PLAN + 1] = bplan[1]
                si[O_PLAN + 2] = bplan[2]
                si[O_COMMIT] = 1
            else:
                break
        np.random.seed(gseed * 1000003 + dec * 7919 + 17)
    st[ST_TOTAL] = sf[F_R]
    st[ST_BFIN] = sf[F_B]
    st[ST_ILEFT] = si[O_I]


@njit(parallel=True, cache=True)
def eval_improved(theta, cfg, n, seed0, K, R, ntri):
    out = np.zeros((n, ST_LEN))
    for g in prange(n):
        improved_game(theta, cfg, K, R, ntri, seed0 + g, out[g])
    return out
