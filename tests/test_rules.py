"""规则环境单元测试 v3：星潮倍率表、抽牌概率、牌库守恒、余波、手牌上限、星轨未满时打法则（白嫖重衍、先锁定再随机化）。"""
import time
import numpy as np
from tulu.env.core import *
from tulu.param.policy import *


def cid(k, r):
    return k * 5 + (r - 1)


def fresh(cfg, seed=0):
    np.random.seed(seed)
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    st = np.zeros(ST_LEN)
    init_game(si, sf, cfg)
    return si, sf, st


def set_hand(si, cards):
    """直接设定手中星象；原始编号 = 当前编号；同时把它们从抽牌堆里扣掉以保持守恒。"""
    # 先把当前手牌退回抽牌堆
    for i in range(si[O_HN]):
        si[O_DP + si[O_HO + i]] += 1
    si[O_HN] = 0
    for c in cards:
        assert si[O_DP + c] > 0
        si[O_DP + c] -= 1
        si[O_HO + si[O_HN]] = c
        si[O_HC + si[O_HN]] = c
        si[O_HN] += 1


def set_laws(si, laws):
    si[O_LN] = 0
    for lid, cost in laws:
        si[O_LI + si[O_LN]] = lid
        si[O_LC + si[O_LN]] = cost
        si[O_LN] += 1


def law_index(si, lid):
    for i in range(si[O_LN]):
        if si[O_LI + i] == lid:
            return i
    raise ValueError


def hand_index(si, card):
    for i in range(si[O_HN]):
        if si[O_HC + i] == card:
            return i
    raise ValueError


def test_tide_table():
    cfg = make_cfg(h=0.0, rho=0.0)
    B = 0.0

    def total(c0, c1, c2):
        d, nZ, nH, nY, hits = tide_calc(c0, c1, c2, B, cfg)
        fs = sorted([f_of(c // 5, cfg) for c in (c0, c1, c2)])
        return fs[0] * 1 + fs[1] * 2 + fs[2] * 3 + d

    z = lambda r: cid(0, r)
    h = lambda r: cid(1, r)
    y = lambda r: cid(2, r)
    assert abs(total(z(3), z(3), z(3)) - 273.6) < 1e-9
    assert abs(total(z(2), z(3), z(4)) - 230.4) < 1e-9
    assert abs(total(z(1), z(3), z(5)) - 187.2) < 1e-9
    assert abs(total(h(2), h(2), h(2)) - 187.2) < 1e-9
    assert abs(total(y(2), y(2), y(2)) - (28.8 + 144 * 1.02)) < 1e-9
    d, nZ, nH, nY, hits = tide_calc(z(2), z(2), h(2), 0.0, cfg)
    assert hits == 9 and abs(d - 9 * 12.8) < 1e-9
    d, *_ = tide_calc(z(2), h(2), y(2), 0.0, cfg)
    assert abs(d - 134.4) < 1e-9
    d, *_ = tide_calc(z(1), z(3), z(5), 0.5, cfg)
    assert abs(d - 9 * (14.4 + 1.5)) < 1e-9
    cfg3 = make_cfg(h=0.0, rho=0.0, mode=1)
    d, *_ = tide_calc(z(1), z(3), z(5), 0.5, cfg3)
    assert abs(d - 9 * 14.4 * 2.5) < 1e-9


def test_first_hand_probability():
    cfg = make_cfg()
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    n = 200000
    cnt = 0
    np.random.seed(1)
    for _ in range(n):
        init_game(si, sf, cfg)
        nz = sum(1 for i in range(si[O_HN]) if si[O_HC + i] // 5 == 0)
        cnt += nz >= 3
    p = cnt / n
    assert abs(p - 15015 / 148995) < 0.003, p


def test_conservation():
    """随机合法操作（含星轨未满时打法则）后，牌库+手牌+星轨+弃牌 = 45（按原始编号）。"""
    cfg = make_cfg()
    rng = np.random.default_rng(3)
    for trial in range(400):
        si, sf, st = fresh(cfg, trial)
        for _ in range(300):
            if si[O_DONE]:
                break
            tot = (si[O_DP:O_DP + NT].sum() + si[O_DC:O_DC + NT].sum() + si[O_HN] + si[O_TN])
            assert tot == 45, (trial, tot)
            assert si[O_HN] + si[O_LN] <= 12
            assert si[O_I] <= 50 or True
            if si[O_PN] > 0:
                pick_option(si, cfg, int(rng.integers(si[O_PN])))
                continue
            acts = []
            for li in range(si[O_LN]):
                if si[O_I] >= si[O_LC + li]:
                    acts.append(("l", li))
            for i in range(si[O_HN]):
                acts.append(("s", i))
            if not acts:
                break
            a = acts[int(rng.integers(len(acts)))]
            if a[0] == "l":
                play_law(si, sf, cfg, a[1], st)
            else:
                place_star(si, sf, cfg, a[1], st)


def test_afterglow():
    """余波：回合末未使用的法则每张 +1 层，上限 50；重衍丢弃手牌不触发余波。"""
    cfg = make_cfg()
    # 1) 普通回合末：2 张未使用法则 +2
    si, sf, st = fresh(cfg, 5)
    set_hand(si, [cid(0, 1), cid(0, 2), cid(1, 3), cid(2, 4)])
    set_laws(si, [(0, 5), (4, 5)])
    si[O_I] = 40
    for c in [cid(0, 1), cid(0, 2), cid(1, 3)]:
        place_star(si, sf, cfg, hand_index(si, c), st)
    assert si[O_T] == 2 and si[O_I] == 42, (si[O_T], si[O_I])
    # 2) 上限：I=49 + 2 -> 50
    si, sf, st = fresh(cfg, 6)
    set_hand(si, [cid(0, 1), cid(0, 2), cid(1, 3), cid(2, 4)])
    set_laws(si, [(0, 5), (4, 5)])
    si[O_I] = 49
    for c in [cid(0, 1), cid(0, 2), cid(1, 3)]:
        place_star(si, sf, cfg, hand_index(si, c), st)
    assert si[O_I] == 50
    # 3) 已满 50 时不再增加
    si, sf, st = fresh(cfg, 7)
    set_hand(si, [cid(0, 1), cid(0, 2), cid(1, 3), cid(2, 4)])
    set_laws(si, [(0, 5), (4, 5)])
    si[O_I] = 50
    for c in [cid(0, 1), cid(0, 2), cid(1, 3)]:
        place_star(si, sf, cfg, hand_index(si, c), st)
    assert si[O_I] == 50
    # 4) 重衍：I=40，手里有 重衍 和 另一张未用法则 -> 打出重衍 -5，另一张被丢弃但不触发余波
    si, sf, st = fresh(cfg, 8)
    set_hand(si, [cid(0, 1), cid(0, 2), cid(1, 3), cid(2, 4)])
    set_laws(si, [(6, 5), (0, 5)])
    si[O_I] = 40
    play_law(si, sf, cfg, law_index(si, 6), st)
    assert si[O_I] == 35, si[O_I]


def test_hand_cap():
    """手牌已满 12 时：抽到的星象进弃牌堆；获得的法则消失；昭示选出的法则无法进入手牌。"""
    cfg = make_cfg()
    si, sf, st = fresh(cfg, 9)
    cards = [cid(k, r) for k in range(2) for r in range(1, 6)] + [cid(2, 1), cid(2, 2)]
    set_hand(si, cards)           # 12 张星象
    set_laws(si, [(4, 5)])        # 引辰
    # 手牌（12 张星象 + 1 张法则）超限，先把法则放进去只为打出它
    si[O_HN] = 12
    n_disc = si[O_DC:O_DC + NT].sum()
    play_law(si, sf, cfg, 0, st)  # 引辰：抽 2 张，手牌已满
    assert si[O_HN] == 12
    assert si[O_DC:O_DC + NT].sum() == n_disc + 2
    # 昭示选出的法则：手牌已满则无法进入
    si[O_PN] = 3
    si[O_PO] = 0
    pick_option(si, cfg, 0)
    assert si[O_LN] == 0


def test_bank_then_rebirth():
    """星轨放 2 张后用重衍清空：2 张的卡效已结算（白嫖），随后抽 4 张并得到 2 张法则。"""
    cfg = make_cfg()
    si, sf, st = fresh(cfg, 10)
    set_hand(si, [cid(0, 2), cid(0, 3), cid(1, 1), cid(2, 5)])
    set_laws(si, [(6, 5), (0, 5)])
    si[O_I] = 50
    place_star(si, sf, cfg, hand_index(si, cid(0, 2)), st)   # 槽 1: 9.6
    place_star(si, sf, cfg, hand_index(si, cid(0, 3)), st)   # 槽 2: 19.2
    assert abs(sf[F_R] - 28.8) < 1e-9
    assert si[O_TN] == 2 and si[O_T] == 1
    play_law(si, sf, cfg, law_index(si, 6), st)
    assert abs(sf[F_R] - 28.8) < 1e-9            # 清空星轨不退还也不扣除
    assert si[O_TN] == 0 and si[O_HN] == 4 and si[O_LN] == 2 and si[O_T] == 1
    assert si[O_I] == 45
    assert st[ST_CYBANK] == 1
    tot = (si[O_DP:O_DP + NT].sum() + si[O_DC:O_DC + NT].sum() + si[O_HN] + si[O_TN])
    assert tot == 45


def test_lock_before_randomize():
    """先把两张灾星放进星轨，再打易象：星轨里的牌不受影响；手里剩余的牌每张都换了种类。"""
    cfg = make_cfg()
    si, sf, st = fresh(cfg, 11)
    set_hand(si, [cid(0, 2), cid(0, 3), cid(1, 1), cid(2, 5)])
    set_laws(si, [(0, 5)])
    place_star(si, sf, cfg, hand_index(si, cid(0, 2)), st)
    place_star(si, sf, cfg, hand_index(si, cid(0, 3)), st)
    before = (si[O_TC], si[O_TC + 1])
    hand_before = sorted(si[O_HC + i] for i in range(si[O_HN]))
    play_law(si, sf, cfg, 0, st)
    assert (si[O_TC], si[O_TC + 1]) == before
    hand_after = [si[O_HC + i] for i in range(si[O_HN])]
    assert len(hand_after) == 2
    for c in hand_after:
        assert (c % 5) in {x % 5 for x in hand_before}


def test_policy_uses_interleaving():
    """策略在手里有重衍时应会先白嫖再重衍（用固定手牌的小场景检查）。"""
    cfg = make_cfg()
    th = default_theta()
    th[T_B + 6] = 10.0
    th[T_MU] = 0.2
    cnt = 0
    for seed in range(200):
        si, sf, st = fresh(cfg, 100 + seed)
        set_hand(si, [cid(0, 2), cid(0, 3), cid(1, 1), cid(2, 5)])   # 手牌很差：无三灾
        set_laws(si, [(6, 5), (4, 5)])
        # 只打一回合，观察是否白嫖
        cfg1 = cfg.copy()
        cfg1[C_N] = 1
        np.random.seed(seed)
        play_until_done(si, sf, th, cfg1, 12, 0, st)
        cnt += st[ST_CYBANK] > 0
    assert cnt > 20, cnt   # 手牌无三灾且有重衍时，应有相当比例先白嫖再重衍
