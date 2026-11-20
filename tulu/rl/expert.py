"""蒸馏的「专家」：滚动前瞻（以方法 A 的参数化策略为基础策略）。

专家在每个决策点对每个候选动作模拟完整局，选平均总倍率最高者。候选动作可能是组合动作
（先锁定放入 1~2 张星象再打法则，或补满星轨），本模块把它们拆成神经网络动作空间里的
「原子动作」序列，并逐步记录（观测、合法掩码、专家动作、剩余回报），用作监督学习的标签。

两种用法：
  * generate_expert_dataset：让专家自己打若干局，记录它走过的状态（行为克隆用）。
  * label_batch：给定任意状态（例如学生策略走到的状态）返回专家在该状态的原子动作（DAgger 用）。
"""
import numpy as np
from numba import njit, prange

from tulu.env.core import *
from tulu.param.policy import *
from tulu.param.rollout import KIND_FINISH, KIND_LAW, KIND_OPTION, action_value
from tulu.rl.vec_env import build_obs_mask, OBS_DIM, NUM_ACTIONS, A_LAW0, A_FREE0, A_OPT0

MAXS = 160          # 每局最多记录的原子步数（超出部分丢弃，极少发生）
RET_SCALE = 100.0   # 剩余回报缩放，与 PPO 的奖励缩放一致


@njit(cache=True)
def law_action(si, li):
    lid = si[O_LI + li]
    if si[O_LC + li] > 0:
        return A_LAW0 + lid
    return A_FREE0 + lid


@njit(cache=True)
def decide(si, sf, theta, cfg, K, R, sb, plan0):
    """滚动前瞻决策。返回 (kind, a, l1, l2)；plan0 写入补满星轨的方案。把当前状态视为一个全新的决策点。"""
    t = int(si[O_T])
    hn = si[O_HN]
    tn = si[O_TN]
    out = np.zeros(2, dtype=np.int64)
    V0 = best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0)
    if si[O_PN] > 0:
        bv = -1e18
        bj = 0
        for j in range(si[O_PN]):
            v = action_value(si, sf, theta, cfg, K, R, sb, KIND_OPTION, j, -1, -1, 0, 0, 0)
            if v > bv:
                bv = v
                bj = j
        return KIND_OPTION, bj, -1, -1
    best = -1e18
    bkind = -1
    ba = 0
    bl1 = -1
    bl2 = -1
    if V0 > -1e17:
        best = action_value(si, sf, theta, cfg, K, R, sb, KIND_FINISH, 0, -1, -1,
                            plan0[0], plan0[1], plan0[2])
        bkind = KIND_FINISH
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
    return bkind, ba, bl1, bl2


@njit(cache=True)
def first_primitive(si, kind, a, l1, l2, plan0):
    """组合动作的第一个原子动作编号（星象动作编号 = 星象当前形态编号）。"""
    if kind == KIND_OPTION:
        return A_OPT0 + si[O_PO + a]
    if kind == KIND_FINISH:
        return plan0[si[O_TN]]
    if l1 >= 0:
        return l1
    if l2 >= 0:
        return l2
    return law_action(si, a)


@njit(cache=True)
def record_step(si, sf, cfg, act, n, obs_b, mask_b, act_b, r_b):
    if n >= MAXS:
        return
    build_obs_mask(si, sf, cfg, obs_b[n], mask_b[n])
    act_b[n] = act
    r_b[n] = sf[F_R]


@njit(cache=True)
def expert_game(theta, cfg, K, R, gseed, obs_b, mask_b, act_b, r_b, ret_b, st):
    """专家打一整局并逐个原子动作记录。返回记录条数。"""
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    np.random.seed(gseed)
    init_game(si, sf, cfg)
    plan0 = np.zeros(3, dtype=np.int64)
    n = 0
    dec = 0
    while si[O_DONE] == 0 and dec < 600:
        dec += 1
        sb = gseed * 7919 + dec * 104729
        kind, a, l1, l2 = decide(si, sf, theta, cfg, K, R, sb, plan0)
        if kind == KIND_OPTION:
            record_step(si, sf, cfg, first_primitive(si, kind, a, l1, l2, plan0), n, obs_b, mask_b, act_b, r_b)
            n += 1
            pick_option(si, cfg, a)
        elif kind == KIND_FINISH:
            while si[O_TN] < 3 and si[O_DONE] == 0:
                c = plan0[si[O_TN]]
                record_step(si, sf, cfg, c, n, obs_b, mask_b, act_b, r_b)
                n += 1
                place_by_value(si, sf, cfg, c, st)
                if si[O_TN] == 0:      # 放满后回合结束，星轨已清空
                    break
        elif kind == KIND_LAW:
            if l1 >= 0:
                record_step(si, sf, cfg, l1, n, obs_b, mask_b, act_b, r_b)
                n += 1
                place_by_value(si, sf, cfg, l1, st)
            if l2 >= 0:
                record_step(si, sf, cfg, l2, n, obs_b, mask_b, act_b, r_b)
                n += 1
                place_by_value(si, sf, cfg, l2, st)
            record_step(si, sf, cfg, law_action(si, a), n, obs_b, mask_b, act_b, r_b)
            n += 1
            play_law(si, sf, cfg, a, st)
        else:
            break
        np.random.seed(gseed * 1000003 + dec * 7919 + 17)
    total = sf[F_R]
    m = min(n, MAXS)
    for i in range(m):
        ret_b[i] = (total - r_b[i]) / RET_SCALE
    st[ST_TOTAL] = total
    return m


@njit(parallel=True, cache=True)
def expert_games_batch(theta, cfg, n_games, seed0, K, R, obs_b, mask_b, act_b, ret_b, cnt, totals):
    for g in prange(n_games):
        r_b = np.zeros(MAXS)
        st = np.zeros(ST_LEN)
        cnt[g] = expert_game(theta, cfg, K, R, seed0 + g, obs_b[g], mask_b[g], act_b[g], r_b, ret_b[g], st)
        totals[g] = st[ST_TOTAL]


@njit(parallel=True, cache=True)
def label_batch(SI, SF, cfg, theta, K, R, seed0, labels):
    """对每个并行环境的当前状态，返回专家的原子动作。不修改环境状态。"""
    for e in prange(SI.shape[0]):
        si = SI[e].copy()
        sf = SF[e].copy()
        si[O_COMMIT] = 0
        plan0 = np.zeros(3, dtype=np.int64)
        np.random.seed(seed0 + e)
        kind, a, l1, l2 = decide(si, sf, theta, cfg, K, R, (seed0 + e) * 7919 + 13, plan0)
        if kind < 0:
            labels[e] = -1
        else:
            labels[e] = first_primitive(si, kind, a, l1, l2, plan0)


def generate_expert_dataset(theta, cfg, n_games, seed0=1_000_000, K=12, R=200, chunk=500, log=print):
    """让专家打 n_games 局并记录。返回 dict(obs float16, mask bool, act int64, ret float32, totals)。"""
    theta = np.asarray(theta, dtype=np.float64)
    cfg = np.asarray(cfg, dtype=np.float64)
    O, M, A, Rt, T = [], [], [], [], []
    done = 0
    while done < n_games:
        n = min(chunk, n_games - done)
        obs_b = np.zeros((n, MAXS, OBS_DIM), dtype=np.float32)
        mask_b = np.zeros((n, MAXS, NUM_ACTIONS), dtype=np.bool_)
        act_b = np.zeros((n, MAXS), dtype=np.int64)
        ret_b = np.zeros((n, MAXS))
        cnt = np.zeros(n, dtype=np.int64)
        totals = np.zeros(n)
        expert_games_batch(theta, cfg, n, seed0 + done, K, R, obs_b, mask_b, act_b, ret_b, cnt, totals)
        for g in range(n):
            c = cnt[g]
            O.append(obs_b[g, :c].astype(np.float16))
            M.append(mask_b[g, :c])
            A.append(act_b[g, :c])
            Rt.append(ret_b[g, :c].astype(np.float32))
        T.append(totals)
        done += n
        log(f"  expert games {done}/{n_games}  (本批平均总倍率 {totals.mean():.1f})")
    return {"obs": np.concatenate(O), "mask": np.concatenate(M), "act": np.concatenate(A),
            "ret": np.concatenate(Rt), "totals": np.concatenate(T)}
