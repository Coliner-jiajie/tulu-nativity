"""诞临·图鲁 诞临阶段规则环境 v3（numba 加速）——所有方法共用。

规则与假设见 ../诞临阶段建模_v0.1.md（A-01 ~ A-32）。伤害全部以「×ATK」为单位记录。

v3 相对 v2 的变化：
  * 余波：回合结束时自动弃置的未使用法则，每张使记忆铭文 +1（上限 imax=50）。重衍丢弃的手牌中的未使用法则同样触发余波（用户确认，2026-11-20）。
  * 手牌已满 12 时，新获得的牌无法进入手牌（星象进弃牌堆，法则消失）；昭示选出的法则同样受限。
  * 法则可以在放第 3 张星象之前的任意时刻打出（星轨可以只放 1~2 张）。策略因此新增：
      - 重衍前先把手里最好的两张星象放进星轨「白嫖」卡效；
      - 易象/增辉/敛芒前先把好牌放进星轨锁定，只让剩余手牌被改动。

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
 C_GROW, C_NDRAW, C_NLAW, C_IMAX, C_AFTER) = range(24)
CFG_LEN = 24


def make_cfg(**kw):
    c = np.zeros(CFG_LEN)
    c[C_N] = 6            # 回合数
    c[C_I0] = 50          # 初始记忆铭文
    c[C_LAWCOST] = 5      # 法则花费
    c[C_CAP] = 12         # 手牌上限
    c[C_FZ], c[C_FH], c[C_FY] = 9.6, 7.2, 4.8
    c[C_BASEHIT] = 9.6
    c[C_PERZ] = 1.6
    c[C_Y3] = 1.02
    c[C_RHO] = 0.20816    # DEF/ATK（60 级面板 + 12 个黄金造物）
    c[C_M] = 1.25         # 铭文对获得触腕伤害的放大
    c[C_H] = 3.137        # H/ATK（L=6）
    c[C_MODE] = 0         # 0: 每次伤害 +kappa*B（加法，已确认）
    c[C_FIRSTGROW] = 0
    c[C_KAPPA] = 3.0      # 300%
    c[C_CARDB] = 0.03
    c[C_TIDEB] = 0.01
    c[C_B0] = 0.05
    c[C_GROW] = 0.0125
    c[C_NDRAW] = 4
    c[C_NLAW] = 2
    c[C_IMAX] = 50        # 记忆铭文上限
    c[C_AFTER] = 1        # 余波：每张未使用法则回合末 +1 层
    names = {"N": C_N, "I0": C_I0, "lawcost": C_LAWCOST, "cap": C_CAP, "fZ": C_FZ,
             "fH": C_FH, "fY": C_FY, "basehit": C_BASEHIT, "perz": C_PERZ, "y3": C_Y3,
             "rho": C_RHO, "m": C_M, "h": C_H, "mode": C_MODE, "firstgrow": C_FIRSTGROW,
             "kappa": C_KAPPA, "cardb": C_CARDB, "tideb": C_TIDEB, "b0": C_B0,
             "grow": C_GROW, "ndraw": C_NDRAW, "nlaw": C_NLAW, "imax": C_IMAX, "after": C_AFTER}
    for k, v in kw.items():
        c[names[k]] = v
    return c


# ---- 状态数组布局 ---------------------------------------------------------
O_HN = 0
O_HO = 1
O_HC = 17
O_LN = 33
O_LI = 34
O_LC = 50
O_TN = 66
O_TC = 67
O_TO = 70
O_DP = 73
O_DC = 88
O_I = 103
O_T = 104
O_PN = 105
O_PO = 106
O_COMMIT = 109
O_DONE = 110
O_PLAN = 111
SI_LEN = 120

F_B = 0
F_R = 1
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
ST_TURNY = 22     # [6]
ST_TURNHITS = 28  # [6]
ST_TURNZ = 34     # [6]
ST_REFUND = 40    # 余波累计返还的铭文层数
ST_CYBANK = 41    # 在星轨已有牌时打出重衍的次数（白嫖）
ST_LOCKLAW = 42   # 在星轨已有牌时打出任意法则的次数
ST_LEN = 48

# ---- 策略参数向量布局 -----------------------------------------------------
T_MU = 0
T_PZ = 1
T_B = 2
T_PHI = 9
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
    """手牌已满时，抽到的星象无法进入手牌，进弃牌堆。"""
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
    """手牌已满时，获得的法则无法进入手牌（法则不进牌库，直接消失）。"""
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
def end_turn(si, sf, cfg, st):
    # 余波：回合结束自动弃置的未使用法则，每张使记忆铭文 +after（上限 imax）
    gain = int(cfg[C_AFTER]) * si[O_LN]
    imax = int(cfg[C_IMAX])
    if gain > 0 and si[O_I] < imax:
        new_i = min(imax, si[O_I] + gain)
        st[ST_REFUND] += new_i - si[O_I]
        si[O_I] = new_i
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
        end_turn(si, sf, cfg, st)


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
    if si[O_TN] > 0:
        st[ST_LOCKLAW] += 1
        if lid == 6:
            st[ST_CYBANK] += 1
    cap = int(cfg[C_CAP])
    if lid == 0:      # 易象：手中每张星象独立随机变成另两种之一
        for i in range(si[O_HN]):
            c = si[O_HC + i]
            nk = ((c // 5) + 1 + np.random.randint(0, 2)) % 3
            si[O_HC + i] = nk * 5 + c % 5
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
    else:             # 重衍：清空星轨，丢弃全部手牌（含其他法则；被丢弃的未使用法则触发余波，每张 +after），抽 4 张，得 2 张法则
        gain = int(cfg[C_AFTER]) * si[O_LN]
        imax = int(cfg[C_IMAX])
        if gain > 0 and si[O_I] < imax:
            new_i = min(imax, si[O_I] + gain)
            st[ST_REFUND] += new_i - si[O_I]
            si[O_I] = new_i
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
def pick_option(si, cfg, j):
    """昭示：把发现的第 j 张法则加入手牌（消耗为 0）；手牌已满则无法进入。"""
    lid = si[O_PO + j]
    add_law(si, lid, 0, int(cfg[C_CAP]))
    si[O_PN] = 0



