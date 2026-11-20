"""Q 值专家：滚动前瞻在每个决策点对各候选动作的价值估计（蒙特卡洛 Q 值），记录成 36 维 Q 向量，
用来训练 DQN 类网络（回归 Q 值，而不只是模仿最优动作）。

与 expert.py 的区别：decide_q 额外输出各候选动作（按它们的第一个原子动作归并）的价值；
同一组合动作的后续原子步没有重新评估，只记录该组合的价值（对应被选中的原子动作）。
未评估的动作用 NaN 表示（保存时）。价值单位：×ATK 的剩余总倍率（训练时除以 100）。"""
import numpy as np
from numba import njit, prange

from tulu.env.core import *
from tulu.param.policy import *
from tulu.param.rollout import KIND_FINISH, KIND_LAW, KIND_OPTION, action_value
from tulu.rl.vec_env import build_obs_mask, OBS_DIM, NUM_ACTIONS, A_OPT0
from tulu.rl.expert import law_action, first_primitive, MAXS, RET_SCALE

NEG = -1e18


@njit(cache=True)
def _upd(qvec, prim, v):
    if v > qvec[prim]:
        qvec[prim] = v


@njit(cache=True)
def decide_q(si, sf, theta, cfg, K, R, ntri, sb, plan0, qvec, meta):
    """同 expert.decide，另把各候选的价值按第一个原子动作写入 qvec（取最大），meta[0] = 最优价值。"""
    t = int(si[O_T])
    hn = si[O_HN]
    tn = si[O_TN]
    out = np.zeros(2, dtype=np.int64)
    plans = np.zeros((8, 3), dtype=np.int64)
    scs = np.zeros(8)
    V0 = best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0)
    if si[O_PN] > 0:
        bv = -1e18
        bj = 0
        for j in range(si[O_PN]):
            v = action_value(si, sf, theta, cfg, K, R, sb, KIND_OPTION, j, -1, -1, 0, 0, 0)
            _upd(qvec, A_OPT0 + si[O_PO + j], v)
            if v > bv:
                bv = v
                bj = j
        meta[0] = bv
        return KIND_OPTION, bj, -1, -1
    best = -1e18
    bkind = -1
    ba = 0
    bl1 = -1
    bl2 = -1
    if V0 > -1e17:
        nk = top_completions(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta,
                             min(ntri, 8), plans, scs)
        bq = 0
        for q in range(nk):
            v = action_value(si, sf, theta, cfg, K, R, sb, KIND_FINISH, 0, -1, -1,
                             plans[q, 0], plans[q, 1], plans[q, 2])
            _upd(qvec, plans[q, tn], v)
            if v > best:
                best = v
                bkind = KIND_FINISH
                bq = q
        if bkind == KIND_FINISH:
            for z in range(3):
                plan0[z] = plans[bq, z]
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
        if l1 >= 0:
            prim = l1
        elif l2 >= 0:
            prim = l2
        else:
            prim = law_action(si, li)
        _upd(qvec, prim, v)
        if v > best:
            best = v
            bkind = KIND_LAW
            ba = li
            bl1 = l1
            bl2 = l2
        if l1 >= 0 or l2 >= 0:
            v = action_value(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, -1, -1, 0, 0, 0)
            _upd(qvec, law_action(si, li), v)
            if v > best:
                best = v
                bkind = KIND_LAW
                ba = li
                bl1 = -1
                bl2 = -1
    meta[0] = best
    return bkind, ba, bl1, bl2


@njit(cache=True)
def record_step_q(si, sf, cfg, act, n, obs_b, mask_b, act_b, r_b, q_b, qrow):
    if n >= MAXS:
        return
    build_obs_mask(si, sf, cfg, obs_b[n], mask_b[n])
    act_b[n] = act
    r_b[n] = sf[F_R]
    for k in range(NUM_ACTIONS):
        q_b[n, k] = qrow[k]


@njit(cache=True)
def expert_game_q(theta, cfg, K, R, ntri, gseed, obs_b, mask_b, act_b, r_b, q_b, ret_b, st):
    si = np.zeros(SI_LEN, dtype=np.int64)
    sf = np.zeros(SF_LEN)
    np.random.seed(gseed)
    init_game(si, sf, cfg)
    plan0 = np.zeros(3, dtype=np.int64)
    qvec = np.full(NUM_ACTIONS, NEG)
    qfol = np.full(NUM_ACTIONS, NEG)
    meta = np.zeros(2)
    n = 0
    dec = 0
    while si[O_DONE] == 0 and dec < 600:
        dec += 1
        sb = gseed * 7919 + dec * 104729
        for k in range(NUM_ACTIONS):
            qvec[k] = NEG
        kind, a, l1, l2 = decide_q(si, sf, theta, cfg, K, R, ntri, sb, plan0, qvec, meta)
        best = meta[0]
        if kind == KIND_OPTION:
            record_step_q(si, sf, cfg, first_primitive(si, kind, a, l1, l2, plan0), n,
                          obs_b, mask_b, act_b, r_b, q_b, qvec)
            n += 1
            pick_option(si, cfg, a)
        elif kind == KIND_FINISH:
            first = True
            while si[O_TN] < 3 and si[O_DONE] == 0:
                c = plan0[si[O_TN]]
                if first:
                    record_step_q(si, sf, cfg, c, n, obs_b, mask_b, act_b, r_b, q_b, qvec)
                else:
                    for k in range(NUM_ACTIONS):
                        qfol[k] = NEG
                    qfol[c] = best
                    record_step_q(si, sf, cfg, c, n, obs_b, mask_b, act_b, r_b, q_b, qfol)
                first = False
                n += 1
                place_by_value(si, sf, cfg, c, st)
                if si[O_TN] == 0:
                    break
        elif kind == KIND_LAW:
            first = True
            if l1 >= 0:
                record_step_q(si, sf, cfg, l1, n, obs_b, mask_b, act_b, r_b, q_b, qvec)
                first = False
                n += 1
                place_by_value(si, sf, cfg, l1, st)
            if l2 >= 0:
                if first:
                    record_step_q(si, sf, cfg, l2, n, obs_b, mask_b, act_b, r_b, q_b, qvec)
                else:
                    for k in range(NUM_ACTIONS):
                        qfol[k] = NEG
                    qfol[l2] = best
                    record_step_q(si, sf, cfg, l2, n, obs_b, mask_b, act_b, r_b, q_b, qfol)
                first = False
                n += 1
                place_by_value(si, sf, cfg, l2, st)
            la = law_action(si, a)
            if first:
                record_step_q(si, sf, cfg, la, n, obs_b, mask_b, act_b, r_b, q_b, qvec)
            else:
                for k in range(NUM_ACTIONS):
                    qfol[k] = NEG
                qfol[la] = best
                record_step_q(si, sf, cfg, la, n, obs_b, mask_b, act_b, r_b, q_b, qfol)
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
def expert_games_batch_q(theta, cfg, n_games, seed0, K, R, ntri, obs_b, mask_b, act_b, q_b, ret_b, rsf_b, cnt, totals):
    for g in prange(n_games):
        st = np.zeros(ST_LEN)
        cnt[g] = expert_game_q(theta, cfg, K, R, ntri, seed0 + g, obs_b[g], mask_b[g], act_b[g], rsf_b[g],
                               q_b[g], ret_b[g], st)
        totals[g] = st[ST_TOTAL]


@njit(parallel=True, cache=True)
def label_batch_q(SI, SF, cfg, theta, K, R, ntri, seed0, labels, qout):
    """对每个并行环境的当前状态：专家的原子动作与完整的 Q 向量（DAgger / 学生状态打标签用）。"""
    for e in prange(SI.shape[0]):
        si = SI[e].copy()
        sf = SF[e].copy()
        si[O_COMMIT] = 0
        plan0 = np.zeros(3, dtype=np.int64)
        qvec = np.full(NUM_ACTIONS, NEG)
        meta = np.zeros(2)
        np.random.seed(seed0 + e)
        kind, a, l1, l2 = decide_q(si, sf, theta, cfg, K, R, ntri, (seed0 + e) * 7919 + 13, plan0, qvec, meta)
        for k in range(NUM_ACTIONS):
            qout[e, k] = qvec[k]
        if kind < 0:
            labels[e] = -1
        else:
            labels[e] = first_primitive(si, kind, a, l1, l2, plan0)


def generate_q_dataset(theta, cfg, n_games, seed0=1_000_000, K=12, R=200, chunk=500, log=print, ntri=1):
    """专家打 n_games 局并记录（观测、掩码、专家动作、36 维 Q 向量、剩余回报）。
    qv 为「剩余回报」（×ATK，已减去该状态前已累计的倍率），未评估的动作为 NaN，与 ret（/100）同口径。"""
    theta = np.asarray(theta, dtype=np.float64)
    cfg = np.asarray(cfg, dtype=np.float64)
    O, M, A, Q, Rt, T = [], [], [], [], [], []
    done = 0
    while done < n_games:
        n = min(chunk, n_games - done)
        obs_b = np.zeros((n, MAXS, OBS_DIM), dtype=np.float32)
        mask_b = np.zeros((n, MAXS, NUM_ACTIONS), dtype=np.bool_)
        act_b = np.zeros((n, MAXS), dtype=np.int64)
        q_b = np.zeros((n, MAXS, NUM_ACTIONS), dtype=np.float32)
        ret_b = np.zeros((n, MAXS))
        rsf_b = np.zeros((n, MAXS))
        cnt = np.zeros(n, dtype=np.int64)
        totals = np.zeros(n)
        expert_games_batch_q(theta, cfg, n, seed0 + done, K, R, ntri, obs_b, mask_b, act_b, q_b, ret_b, rsf_b,
                             cnt, totals)
        for g in range(n):
            c = cnt[g]
            O.append(obs_b[g, :c].astype(np.float16))
            M.append(mask_b[g, :c])
            A.append(act_b[g, :c])
            q = q_b[g, :c].copy()
            bad = q < -1e17
            q = q - rsf_b[g, :c, None].astype(np.float32)      # 滚动前瞻给的是整局总倍率；减去该状态前已累计的倍率 = 剩余回报
            q[bad] = np.nan
            Q.append(q)
            Rt.append(ret_b[g, :c].astype(np.float32))
        T.append(totals)
        done += n
        log(f"  Q-expert games {done}/{n_games}  (本批平均总倍率 {totals.mean():.1f})")
    return {"obs": np.concatenate(O), "mask": np.concatenate(M), "act": np.concatenate(A),
            "qv": np.concatenate(Q), "ret": np.concatenate(Rt), "totals": np.concatenate(T)}
