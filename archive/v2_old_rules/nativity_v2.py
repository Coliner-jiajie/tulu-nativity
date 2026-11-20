"""诞临·图鲁 诞临阶段仿真环境 + 参数化策略（numba 加速）。

规则与假设见 ../诞临阶段建模_v0.1.md（v0.2）。
伤害全部以「×ATK」为单位记录，只关注倍率。

星象编号：id = k*5 + (r-1)，k: 0=灾 1=垣 2=蕴，r: 1..5。
法则编号：0 易象 1 增辉 2 敛芒 3 昭示 4 引辰 5 拓途 6 重衍。
"""
import numpy as np
from numba import njit, prange

NT = 15
NL = 7
LAW_NAMES = ["易象", "增辉", "敛芒", "昭示", "引辰", "拓途", "重衍"]

# ---- 配置向量下标 -------------------------------------------------------
(C_N, C_I0, C_LAWCOST, C_CAP, C_FZ, C_FH, C_FY, C_BASEHIT, C_PERZ, C_Y3,
 C_RHO, C_M, C_H, C_MODE, C_FIRSTGROW, C_KAPPA, C_CARDB, C_TIDEB, C_B0,
 C_GROW, C_NDRAW, C_NLAW) = range(22)


def make_cfg(**kw):
    c = np.zeros(22)
    c[C_N] = 6            # 回合数
    c[C_I0] = 50          # 初始记忆铭文
    c[C_LAWCOST] = 5      # 法则花费（A-01：无返还）
    c[C_CAP] = 12         # 手牌上限（A-08）
    c[C_FZ], c[C_FH], c[C_FY] = 9.6, 7.2, 4.8
    c[C_BASEHIT] = 9.6
    c[C_PERZ] = 1.6
    c[C_Y3] = 1.02
    c[C_RHO] = 0.56       # DEF/ATK
    c[C_M] = 1.25         # 铭文对获得触腕伤害的放大（A-09）
    c[C_H] = 5.0          # H/ATK
    c[C_MODE] = 0         # 0: 每次伤害 +kappa*B（加法）；1: 每次伤害 ×(1+kappa*B)
    c[C_FIRSTGROW] = 0    # A-13
    c[C_KAPPA] = 3.0      # 300%
    c[C_CARDB] = 0.03
    c[C_TIDEB] = 0.01
    c[C_B0] = 0.05
    c[C_GROW] = 0.0125
    c[C_NDRAW] = 4
    c[C_NLAW] = 2
    names = {"N": C_N, "I0": C_I0, "lawcost": C_LAWCOST, "cap": C_CAP, "fZ": C_FZ,
             "fH": C_FH, "fY": C_FY, "basehit": C_BASEHIT, "perz": C_PERZ, "y3": C_Y3,
             "rho": C_RHO, "m": C_M, "h": C_H, "mode": C_MODE, "firstgrow": C_FIRSTGROW,
             "kappa": C_KAPPA, "cardb": C_CARDB, "tideb": C_TIDEB, "b0": C_B0,
             "grow": C_GROW, "ndraw": C_NDRAW, "nlaw": C_NLAW}
    for k, v in kw.items():
        c[names[k]] = v
    return c


# ---- 状态数组布局 ---------------------------------------------------------
O_HN = 0          # 手中星象数
O_HO = 1          # 手中星象原始编号 [16]
O_HC = 17         # 手中星象当前编号 [16]（法则临时改动后的形态）
O_LN = 33         # 手中法则数
O_LI = 34         # 法则编号 [16]
O_LC = 50         # 法则花费 [16]
O_TN = 66         # 星轨已放张数
O_TC = 67         # 星轨当前形态 [3]
O_TO = 70         # 星轨原始编号 [3]
O_DP = 73         # 抽牌堆各编号张数 [15]
O_DC = 88         # 弃牌堆各编号张数 [15]
O_I = 103         # 记忆铭文
O_T = 104         # 当前回合
O_PN = 105        # 待选发现项数
O_PO = 106        # 发现项 [3]
O_COMMIT = 109    # 已决定放牌
O_DONE = 110
O_PLAN = 111      # 计划放入星轨的三张 [3]（当前形态编号）
SI_LEN = 120

F_B = 0           # 触腕伤害值 B
F_R = 1           # 累计倍率
SF_LEN = 4

# ---- 统计向量布局 ---------------------------------------------------------
ST_TOTAL = 0
ST_NLAW = 1
ST_LAWCNT = 2     # [7]
ST_TURNR = 9      # [6]
ST_N3Z = 15
ST_N3Y = 16
ST_N3H = 17
ST_BFIN = 18
ST_ILEFT = 19
ST_TURNY = 22     # [6] 每回合星轨中蕴星张数
ST_TURNHITS = 28  # [6]
ST_TURNZ = 34     # [6] 每回合星轨中灾星张数
ST_LEN = 40

# ---- 策略参数向量布局 -----------------------------------------------------
T_MU = 0          # 每 1 层铭文的影子价格
T_PZ = 1          # 在非三灾回合里消耗灾星的惩罚
T_B = 2           # 各法则的额外价值 [7]
T_PHI = 9         # 各回合 B 的未来边际价值 φ_t [6]
THETA_LEN = 15


@njit(cache=True)
def f_of(k, cfg):
    if k == 0:
        return cfg[C_FZ]
    elif k == 1:
        return cfg[C_FH]
    return cfg[C_FY]


@njit(cache=True)
def tide_calc(c0, c1, c2, B, cfg):
    """星潮奔涌：返回 (伤害, nZ, nH, nY, 次数)。"""
    k0 = c0 // 5
    k1 = c1 // 5
    k2 = c2 // 5
    r0 = c0 % 5
    r1 = c1 % 5
    r2 = c2 % 5
    hits = 3
    if r0 == r1 and r1 == r2:
        hits += 6
    else:
        lo = min(r0, min(r1, r2))
        hi = max(r0, max(r1, r2))
        mid = r0 + r1 + r2 - lo - hi
        if mid - lo == 1 and hi - mid == 1:
            hits += 3
    if k0 == k1 and k1 == k2:
        hits += 6
    elif k0 != k1 and k1 != k2 and k0 != k2:
        hits += 3
    nZ = int(k0 == 0) + int(k1 == 0) + int(k2 == 0)
    nH = int(k0 == 1) + int(k1 == 1) + int(k2 == 1)
    nY = int(k0 == 2) + int(k1 == 2) + int(k2 == 2)
    per = cfg[C_BASEHIT] + cfg[C_PERZ] * nZ
    if nZ == 3:
        if cfg[C_MODE] < 0.5:
            per += cfg[C_KAPPA] * B
        else:
            per *= 1.0 + cfg[C_KAPPA] * B
    dmg = hits * per
    if nY == 3:
        dmg *= cfg[C_Y3]
    return dmg, nZ, nH, nY, hits


# ---- 抽牌与手牌操作 -------------------------------------------------------
@njit(cache=True)
def draw_one(si):
    tot = 0
    for c in range(NT):
        tot += si[O_DP + c]
    if tot == 0:
        for c in range(NT):
            si[O_DP + c] = si[O_DC + c]
            si[O_DC + c] = 0
            tot += si[O_DP + c]
        if tot == 0:
            return -1
    r = np.random.randint(0, tot)
    for c in range(NT):
        r -= si[O_DP + c]
        if r < 0:
            si[O_DP + c] -= 1
            return c
    return -1


@njit(cache=True)
def add_star(si, c, cap):
    if c < 0:
        return
    n = si[O_HN]
    if n + si[O_LN] < cap:
        si[O_HO + n] = c
        si[O_HC + n] = c
        si[O_HN] = n + 1
    else:
        si[O_DC + c] += 1


@njit(cache=True)
def add_law(si, lid, cost, cap):
    n = si[O_LN]
    if si[O_HN] + n < cap:
        si[O_LI + n] = lid
        si[O_LC + n] = cost
        si[O_LN] = n + 1


@njit(cache=True)
def pick_distinct(n_total, n_pick, out):
    arr = np.arange(n_total)
    for i in range(n_pick):
        j = i + np.random.randint(0, n_total - i)
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
        out[i] = arr[i]


@njit(cache=True)
def gen_laws(si, cfg, n):
    out = np.zeros(8, dtype=np.int64)
    pick_distinct(NL, n, out)
    for i in range(n):
        add_law(si, out[i], int(cfg[C_LAWCOST]), int(cfg[C_CAP]))


@njit(cache=True)
def start_turn(si, sf, cfg):
    t = si[O_T]
    if t > 1 or cfg[C_FIRSTGROW] > 0.5:
        sf[F_B] += cfg[C_GROW] * cfg[C_H] * cfg[C_M]
    si[O_HN] = 0
    si[O_LN] = 0
    si[O_TN] = 0
    si[O_COMMIT] = 0
    si[O_PN] = 0
    cap = int(cfg[C_CAP])
    for _ in range(int(cfg[C_NDRAW])):
        add_star(si, draw_one(si), cap)
    gen_laws(si, cfg, int(cfg[C_NLAW]))


@njit(cache=True)
def end_turn(si, sf, cfg):
    for i in range(si[O_HN]):
        si[O_DC + si[O_HO + i]] += 1
    for i in range(si[O_TN]):
        si[O_DC + si[O_TO + i]] += 1
    si[O_HN] = 0
    si[O_LN] = 0
    si[O_TN] = 0
    si[O_T] += 1
    if si[O_T] > int(cfg[C_N]):
        si[O_DONE] = 1
    else:
        start_turn(si, sf, cfg)


@njit(cache=True)
def init_game(si, sf, cfg):
    si[:] = 0
    sf[:] = 0.0
    for c in range(NT):
        si[O_DP + c] = 3
    si[O_I] = int(cfg[C_I0])
    si[O_T] = 1
    sf[F_B] = cfg[C_B0] * cfg[C_H]
    start_turn(si, sf, cfg)


@njit(cache=True)
def place_star(si, sf, cfg, hi, st):
    """把手中第 hi 张星象放入下一个星轨位置，结算卡效；放满则发动星潮并结束回合。"""
    cur = si[O_HC + hi]
    orig = si[O_HO + hi]
    n = si[O_HN]
    si[O_HC + hi] = si[O_HC + n - 1]
    si[O_HO + hi] = si[O_HO + n - 1]
    si[O_HN] = n - 1
    j = si[O_TN] + 1
    k = cur // 5
    sf[F_R] += f_of(k, cfg) * j
    st[ST_TURNR + si[O_T] - 1] += f_of(k, cfg) * j
    if k == 2:
        sf[F_B] += cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * j
    si[O_TC + si[O_TN]] = cur
    si[O_TO + si[O_TN]] = orig
    si[O_TN] = j
    if j == 3:
        dmg, nZ, nH, nY, hits = tide_calc(si[O_TC], si[O_TC + 1], si[O_TC + 2], sf[F_B], cfg)
        sf[F_R] += dmg
        t = si[O_T] - 1
        st[ST_TURNR + t] += dmg
        st[ST_TURNY + t] = nY
        st[ST_TURNZ + t] = nZ
        st[ST_TURNHITS + t] = hits
        if nZ == 3:
            st[ST_N3Z] += 1
        if nH == 3:
            st[ST_N3H] += 1
        if nY == 3:
            st[ST_N3Y] += 1
        sf[F_B] += cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY
        end_turn(si, sf, cfg)


@njit(cache=True)
def play_law(si, sf, cfg, li, st):
    lid = si[O_LI + li]
    cost = si[O_LC + li]
    si[O_I] -= cost
    n = si[O_LN]
    si[O_LI + li] = si[O_LI + n - 1]
    si[O_LC + li] = si[O_LC + n - 1]
    si[O_LN] = n - 1
    st[ST_NLAW] += 1
    st[ST_LAWCNT + lid] += 1
    cap = int(cfg[C_CAP])
    if lid == 0:      # 易象
        for i in range(si[O_HN]):
            c = si[O_HC + i]
            k = c // 5
            r = c % 5
            nk = (k + 1 + np.random.randint(0, 2)) % 3
            si[O_HC + i] = nk * 5 + r
    elif lid == 1:    # 增辉
        for i in range(si[O_HN]):
            c = si[O_HC + i]
            si[O_HC + i] = (c // 5) * 5 + min(4, c % 5 + 1)
    elif lid == 2:    # 敛芒
        for i in range(si[O_HN]):
            c = si[O_HC + i]
            si[O_HC + i] = (c // 5) * 5 + max(0, c % 5 - 1)
    elif lid == 3:    # 昭示：发现 3 张法则
        out = np.zeros(8, dtype=np.int64)
        pick_distinct(NL, 3, out)
        for i in range(3):
            si[O_PO + i] = out[i]
        si[O_PN] = 3
    elif lid == 4:    # 引辰：抽 2 张星象
        add_star(si, draw_one(si), cap)
        add_star(si, draw_one(si), cap)
    elif lid == 5:    # 拓途：随机获得 3 张不同法则
        out = np.zeros(8, dtype=np.int64)
        pick_distinct(NL, 3, out)
        for i in range(3):
            add_law(si, out[i], int(cfg[C_LAWCOST]), cap)
    else:             # 重衍：清空星轨，丢弃全部手牌（含其他法则），抽 4 张，得 2 张法则
        for i in range(si[O_TN]):
            si[O_DC + si[O_TO + i]] += 1
        si[O_TN] = 0
        for i in range(si[O_HN]):
            si[O_DC + si[O_HO + i]] += 1
        si[O_HN] = 0
        si[O_LN] = 0
        for _ in range(4):
            add_star(si, draw_one(si), cap)
        gen_laws(si, cfg, 2)


@njit(cache=True)
def pick_option(si, j):
    lid = si[O_PO + j]
    # 昭示选出的法则消耗为 0
    n = si[O_LN]
    if si[O_HN] + n < 16:
        si[O_LI + n] = lid
        si[O_LC + n] = 0
        si[O_LN] = n + 1
    si[O_PN] = 0


# ---- 策略 -----------------------------------------------------------------
@njit(cache=True)
def eval_triple(ca, cb, cc, B, t, cfg, theta):
    """三张牌放入空星轨的得分；返回 (score, 按槽位 1/2/3 排序后的牌)。"""
    fa = f_of(ca // 5, cfg)
    fb = f_of(cb // 5, cfg)
    fc = f_of(cc // 5, cfg)
    # 升序排序（f 大的放在触发次数更多的位置）
    s1 = ca
    s2 = cb
    s3 = cc
    f1 = fa
    f2 = fb
    f3 = fc
    if f1 > f2:
        s1, s2 = s2, s1
        f1, f2 = f2, f1
    if f2 > f3:
        s2, s3 = s3, s2
        f2, f3 = f3, f2
    if f1 > f2:
        s1, s2 = s2, s1
        f1, f2 = f2, f1
    card = f1 * 1 + f2 * 2 + f3 * 3
    dmg, nZ, nH, nY, hits = tide_calc(s1, s2, s3, B, cfg)
    dB = 0.0
    if s1 // 5 == 2:
        dB += cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * 1
    if s2 // 5 == 2:
        dB += cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * 2
    if s3 // 5 == 2:
        dB += cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * 3
    dB += cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY
    score = card + dmg + theta[T_PHI + t - 1] * dB
    if nZ < 3:
        score -= theta[T_PZ] * nZ
    return score, s1, s2, s3


@njit(cache=True)
def best_commit(hc, hn, B, t, cfg, theta):
    best = -1e18
    b1 = -1
    b2 = -1
    b3 = -1
    for i in range(hn):
        for j in range(i + 1, hn):
            for k in range(j + 1, hn):
                sc, s1, s2, s3 = eval_triple(hc[i], hc[j], hc[k], B, t, cfg, theta)
                if sc > best:
                    best = sc
                    b1 = s1
                    b2 = s2
                    b3 = s3
    return best, b1, b2, b3


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
def law_value(lid, cost, si, hn, B, t, cfg, theta, K, S0):
    """打出一张法则后（再按最佳三连放牌）的预期得分，含该法则的额外价值与铭文花费。"""
    hc = si[O_HC:O_HC + 16]
    tmp = np.zeros(16, dtype=np.int64)
    dpc = np.zeros(NT, dtype=np.int64)
    base = theta[T_B + lid] - theta[T_MU] * cost
    room = int(cfg[C_CAP]) - (hn + si[O_LN] - 1)
    if lid == 3 or lid == 5:
        return S0 + base
    if lid == 1 or lid == 2:
        for i in range(hn):
            c = hc[i]
            if lid == 1:
                tmp[i] = (c // 5) * 5 + min(4, c % 5 + 1)
            else:
                tmp[i] = (c // 5) * 5 + max(0, c % 5 - 1)
        return best_commit(tmp, hn, B, t, cfg, theta)[0] + base
    tot = 0.0
    for s in range(K):
        if lid == 0:
            for i in range(hn):
                c = hc[i]
                nk = ((c // 5) + 1 + np.random.randint(0, 2)) % 3
                tmp[i] = nk * 5 + c % 5
            n2 = hn
        elif lid == 4:
            for i in range(hn):
                tmp[i] = hc[i]
            for c in range(NT):
                dpc[c] = si[O_DP + c]
            n2 = hn
            for _ in range(2):
                d = sample_draw(dpc)
                if d >= 0 and room > 0:
                    tmp[n2] = d
                    n2 += 1
                    room -= 1
            room = int(cfg[C_CAP]) - (hn + si[O_LN] - 1)
        else:  # 重衍
            for c in range(NT):
                dpc[c] = si[O_DP + c]
            # 重衍会把弃牌堆洗回仅在抽牌堆为空时；这里忽略该罕见情形
            n2 = 0
            for _ in range(4):
                d = sample_draw(dpc)
                if d >= 0:
                    tmp[n2] = d
                    n2 += 1
        if n2 >= 3:
            tot += best_commit(tmp, n2, B, t, cfg, theta)[0]
        else:
            tot += -1e6
    return tot / K + base


@njit(cache=True)
def play_until_done(si, sf, theta, cfg, K, mode, st):
    """从任意状态按策略打到游戏结束。mode 0: 参数化策略；mode 1: 均匀随机合法动作。"""
    guard = 0
    while si[O_DONE] == 0 and guard < 400:
        guard += 1
        t = int(si[O_T])
        if mode == 1:
            # 随机：在可用法则与星象之间均匀选择
            if si[O_PN] > 0:
                pick_option(si, np.random.randint(0, si[O_PN]))
                continue
            nl = 0
            for li in range(si[O_LN]):
                if si[O_I] >= si[O_LC + li]:
                    nl += 1
            ns = si[O_HN]
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
        if si[O_PN] > 0:
            hn = si[O_HN]
            S0 = best_commit(si[O_HC:O_HC + 16], hn, sf[F_B], t, cfg, theta)[0]
            bv = -1e18
            bj = 0
            for j in range(si[O_PN]):
                v = law_value(si[O_PO + j], 0, si, hn, sf[F_B], t, cfg, theta, K, S0)
                if v > bv:
                    bv = v
                    bj = j
            pick_option(si, bj)
            continue
        if si[O_COMMIT] == 1:
            pc = si[O_PLAN + si[O_TN]]
            for i in range(si[O_HN]):
                if si[O_HC + i] == pc:
                    place_star(si, sf, cfg, i, st)
                    break
            continue
        hn = si[O_HN]
        S0, s1, s2, s3 = best_commit(si[O_HC:O_HC + 16], hn, sf[F_B], t, cfg, theta)
        best = S0
        bestl = -1
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
            v = law_value(lid, cost, si, hn, sf[F_B], t, cfg, theta, K, S0)
            if v > best + 1e-9:
                best = v
                bestl = li
        if bestl >= 0:
            play_law(si, sf, cfg, bestl, st)
        else:
            si[O_PLAN] = s1
            si[O_PLAN + 1] = s2
            si[O_PLAN + 2] = s3
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
