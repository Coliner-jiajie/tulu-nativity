"""方法 A：参数化策略（打分公式 + 一步前瞻）。

策略对每个决策点给「补满星轨」与「打某张法则（可先锁定若干星象）」打分，选最高者。
参数向量 θ（15 维）由 CEM 在仿真中搜索，见 tulu/param/cem.py。
"""
import numpy as np
from numba import njit, prange
from tulu.env.core import *

# ---- 策略：三连得分 --------------------------------------------------------
@njit(cache=True)
def score_final(c0, c1, c2, tn, B, t, cfg, theta):
    """最终三张牌 (c0,c1,c2)（前 tn 张已在星轨）的得分：尚未结算的卡效 + 星潮 + 触腕价值 − 灾星稀缺惩罚。"""
    cb = cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M]
    card = 0.0
    dB = 0.0
    if tn <= 0:
        card += f_of(c0 // 5, cfg) * 1
        if c0 // 5 == 2:
            dB += cb * 1
    if tn <= 1:
        card += f_of(c1 // 5, cfg) * 2
        if c1 // 5 == 2:
            dB += cb * 2
    card += f_of(c2 // 5, cfg) * 3
    if c2 // 5 == 2:
        dB += cb * 3
    dmg, nZ, nH, nY, hits = tide_calc(c0, c1, c2, B, cfg)
    dB += cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY
    score = card + dmg + theta[T_PHI + t - 1] * dB
    if nZ < 3:
        score -= theta[T_PZ] * nZ
    return score


@njit(cache=True)
def best_completion(hc, hn, trc, tn, B, t, cfg, theta, plan):
    """已有 tn 张牌在星轨时，用手牌补满的最佳方案；plan[槽位] 写入补牌（升序 f，f 大的放后面）。"""
    s = 3 - tn
    best = -1e18
    if hn < s:
        return best
    if s == 3:
        for i in range(hn):
            for j in range(i + 1, hn):
                for k in range(j + 1, hn):
                    a = hc[i]
                    b = hc[j]
                    c = hc[k]
                    fa = f_of(a // 5, cfg)
                    fb = f_of(b // 5, cfg)
                    fc = f_of(c // 5, cfg)
                    if fa > fb:
                        a, b = b, a
                        fa, fb = fb, fa
                    if fb > fc:
                        b, c = c, b
                        fb, fc = fc, fb
                    if fa > fb:
                        a, b = b, a
                        fa, fb = fb, fa
                    sc = score_final(a, b, c, 0, B, t, cfg, theta)
                    if sc > best:
                        best = sc
                        plan[0] = a
                        plan[1] = b
                        plan[2] = c
    elif s == 2:
        p0 = trc[0]
        for i in range(hn):
            for j in range(i + 1, hn):
                a = hc[i]
                b = hc[j]
                if f_of(a // 5, cfg) > f_of(b // 5, cfg):
                    a, b = b, a
                sc = score_final(p0, a, b, 1, B, t, cfg, theta)
                if sc > best:
                    best = sc
                    plan[1] = a
                    plan[2] = b
    else:
        p0 = trc[0]
        p1 = trc[1]
        for i in range(hn):
            sc = score_final(p0, p1, hc[i], 2, B, t, cfg, theta)
            if sc > best:
                best = sc
                plan[2] = hc[i]
    return best


@njit(cache=True)
def sample_draw(dpc):
    tot = 0
    for c in range(NT):
        tot += dpc[c]
    if tot == 0:
        return -1
    r = np.random.randint(0, tot)
    for c in range(NT):
        r -= dpc[c]
        if r < 0:
            dpc[c] -= 1
            return c
    return -1


@njit(cache=True)
def lock_value(card, slot, cfg, theta, t):
    """把一张星象放进星轨第 slot 位立即获得的价值（卡效 + 蕴星触腕价值）。"""
    v = f_of(card // 5, cfg) * slot
    if card // 5 == 2:
        v += theta[T_PHI + t - 1] * cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * slot
    return v


@njit(cache=True)
def law_value2(lid, cost, si, B, t, cfg, theta, K, V0, out):
    """打出一张法则（可先把若干星象放进星轨）后的预期价值。out[0], out[1] 为先锁定放入的星象（-1 表示无），按放置顺序。"""
    out[0] = -1
    out[1] = -1
    hn = si[O_HN]
    tn = si[O_TN]
    hc = si[O_HC:O_HC + 16]
    trc = si[O_TC:O_TC + 3]
    base = theta[T_B + lid] - theta[T_MU] * cost
    plan = np.zeros(3, dtype=np.int64)
    tmp = np.zeros(16, dtype=np.int64)
    trc2 = np.zeros(3, dtype=np.int64)
    dpc = np.zeros(NT, dtype=np.int64)
    cap = int(cfg[C_CAP])
    room0 = cap - (hn + si[O_LN] - 1)

    if lid == 3 or lid == 5:
        return V0 + base

    if lid == 4:      # 引辰：不锁定
        tot = 0.0
        for s in range(K):
            for i in range(hn):
                tmp[i] = hc[i]
            for c in range(NT):
                dpc[c] = si[O_DP + c]
            n2 = hn
            room = room0
            for _ in range(2):
                d = sample_draw(dpc)
                if d >= 0 and room > 0:
                    tmp[n2] = d
                    n2 += 1
                    room -= 1
            v = best_completion(tmp, n2, trc, tn, B, t, cfg, theta, plan)
            if v < -1e17:
                v = -1e6
            tot += v
        return tot / K + base

    if lid == 6:      # 重衍：先把最好的 (2-tn) 张星象放进星轨白嫖卡效，再清空星轨
        nb = 2 - tn
        lockval = 0.0
        # 找 f 最大与次大
        i1 = -1
        i2 = -1
        for i in range(hn):
            f = f_of(hc[i] // 5, cfg)
            if i1 < 0 or f > f_of(hc[i1] // 5, cfg):
                i2 = i1
                i1 = i
            elif i2 < 0 or f > f_of(hc[i2] // 5, cfg):
                i2 = i
        if nb == 2 and i1 >= 0 and i2 >= 0:
            lockval = lock_value(hc[i2], 1, cfg, theta, t) + lock_value(hc[i1], 2, cfg, theta, t)
            out[0] = hc[i2]
            out[1] = hc[i1]
        elif nb == 2 and i1 >= 0:
            lockval = lock_value(hc[i1], 1, cfg, theta, t)
            out[0] = hc[i1]
        elif nb == 1 and i1 >= 0:
            lockval = lock_value(hc[i1], 2, cfg, theta, t)
            out[0] = hc[i1]
        tot = 0.0
        for s in range(K):
            for c in range(NT):
                dpc[c] = si[O_DP + c]
            n2 = 0
            for _ in range(4):
                d = sample_draw(dpc)
                if d >= 0:
                    tmp[n2] = d
                    n2 += 1
            v = best_completion(tmp, n2, trc2, 0, B, t, cfg, theta, plan)
            if v < -1e17:
                v = -1e6
            tot += v
        return lockval + tot / K + base

    # 易象 / 增辉 / 敛芒：枚举「先锁定若干张」的方案
    best_val = -1e18
    hand2 = np.zeros(16, dtype=np.int64)
    nlock_max = 2 - tn
    nopt = 1
    if nlock_max >= 1:
        nopt += hn
    if nlock_max >= 2:
        nopt += hn * (hn - 1)
    for opt in range(nopt):
        # 解码 opt -> (a, b)
        a = -1
        b = -1
        if opt >= 1 and opt <= hn and nlock_max >= 1:
            a = opt - 1
        elif opt > hn:
            q = opt - 1 - hn
            a = q // (hn - 1)
            r = q % (hn - 1)
            b = r if r < a else r + 1
        lockval = 0.0
        n2 = 0
        for i in range(3):
            trc2[i] = trc[i]
        tn2 = tn
        if a >= 0:
            lockval += lock_value(hc[a], tn2 + 1, cfg, theta, t)
            trc2[tn2] = hc[a]
            tn2 += 1
        if b >= 0:
            lockval += lock_value(hc[b], tn2 + 1, cfg, theta, t)
            trc2[tn2] = hc[b]
            tn2 += 1
        for i in range(hn):
            if i != a and i != b:
                hand2[n2] = hc[i]
                n2 += 1
        if lid == 1 or lid == 2:
            for i in range(n2):
                c = hand2[i]
                if lid == 1:
                    tmp[i] = (c // 5) * 5 + min(4, c % 5 + 1)
                else:
                    tmp[i] = (c // 5) * 5 + max(0, c % 5 - 1)
            v = best_completion(tmp, n2, trc2, tn2, B, t, cfg, theta, plan)
            if v < -1e17:
                continue
            val = lockval + v + base
        else:
            tot = 0.0
            ok = True
            for s in range(K):
                for i in range(n2):
                    c = hand2[i]
                    nk = ((c // 5) + 1 + np.random.randint(0, 2)) % 3
                    tmp[i] = nk * 5 + c % 5
                v = best_completion(tmp, n2, trc2, tn2, B, t, cfg, theta, plan)
                if v < -1e17:
                    ok = False
                    break
                tot += v
            if not ok:
                continue
            val = lockval + tot / K + base
        if val > best_val:
            best_val = val
            out[0] = hc[a] if a >= 0 else -1
            out[1] = hc[b] if b >= 0 else -1
    return best_val


@njit(cache=True)
def place_by_value(si, sf, cfg, cur, st):
    """把手中当前形态为 cur 的一张星象放入星轨。"""
    for i in range(si[O_HN]):
        if si[O_HC + i] == cur:
            place_star(si, sf, cfg, i, st)
            return True
    return False


@njit(cache=True)
def play_until_done(si, sf, theta, cfg, K, mode, st):
    """从任意状态按策略打到游戏结束。mode 0: 参数化策略；mode 1: 均匀随机合法动作。"""
    guard = 0
    plan = np.zeros(3, dtype=np.int64)
    plan0 = np.zeros(3, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    outb = np.zeros(2, dtype=np.int64)
    while si[O_DONE] == 0 and guard < 2000:
        guard += 1
        t = int(si[O_T])
        if mode == 1:
            if si[O_PN] > 0:
                pick_option(si, cfg, np.random.randint(0, si[O_PN]))
                continue
            nl = 0
            for li in range(si[O_LN]):
                if si[O_I] >= si[O_LC + li]:
                    nl += 1
            ns = si[O_HN]
            if nl + ns == 0:
                break
            r = np.random.randint(0, nl + ns)
            if r < nl:
                cnt = 0
                for li in range(si[O_LN]):
                    if si[O_I] >= si[O_LC + li]:
                        if cnt == r:
                            play_law(si, sf, cfg, li, st)
                            break
                        cnt += 1
            else:
                place_star(si, sf, cfg, r - nl, st)
            continue
        hn = si[O_HN]
        tn = si[O_TN]
        V0 = best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0)
        if si[O_PN] > 0:
            bv = -1e18
            bj = 0
            for j in range(si[O_PN]):
                v = law_value2(si[O_PO + j], 0, si, sf[F_B], t, cfg, theta, K, V0, out)
                if v > bv:
                    bv = v
                    bj = j
            pick_option(si, cfg, bj)
            continue
        if si[O_COMMIT] == 1:
            pc = si[O_PLAN + si[O_TN]]
            place_by_value(si, sf, cfg, pc, st)
            continue
        best = V0
        bestl = -1
        outb[0] = -1
        outb[1] = -1
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
            v = law_value2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out)
            if v > best + 1e-9:
                best = v
                bestl = li
                outb[0] = out[0]
                outb[1] = out[1]
        if bestl >= 0:
            for q in range(2):
                if outb[q] >= 0:
                    place_by_value(si, sf, cfg, outb[q], st)
            play_law(si, sf, cfg, bestl, st)
        else:
            if V0 < -1e17:
                break
            si[O_PLAN] = plan0[0]
            si[O_PLAN + 1] = plan0[1]
            si[O_PLAN + 2] = plan0[2]
            si[O_COMMIT] = 1
    st[ST_TOTAL] = sf[F_R]
    st[ST_BFIN] = sf[F_B]
    st[ST_ILEFT] = si[O_I]


@njit(cache=True)
def run_game(theta, cfg, K, mode, st):
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    init_game(si, sf, cfg)
    play_until_done(si, sf, theta, cfg, K, mode, st)


@njit(parallel=True, cache=True)
def eval_policy(theta, cfg, n, seed0, K, mode):
    out = np.zeros((n, ST_LEN))
    for g in prange(n):
        np.random.seed(seed0 + g)
        run_game(theta, cfg, K, mode, out[g])
    return out


def default_theta():
    th = np.zeros(THETA_LEN)
    th[T_MU] = 1.0
    th[T_PZ] = 3.0
    th[T_PHI:T_PHI + 6] = [150, 120, 90, 60, 30, 0]
    return th


def no_law_theta():
    th = default_theta()
    th[T_MU] = 1e6
    th[T_PZ] = 0.0
    th[T_PHI:T_PHI + 6] = 0
    return th


def greedy_theta():
    th = np.zeros(THETA_LEN)
    return th
