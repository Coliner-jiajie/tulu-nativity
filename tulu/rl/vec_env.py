"""方法 B：向量化环境（numba）。同时推进 N 个独立对局，动作统一为 36 维离散空间并带合法动作掩码。

动作编号（NUM_ACTIONS = 36）：
    0..14   放入当前形态为 (种类, 星阶) 的一张星象（编号 = 种类*5 + 星阶-1）
    15..21  打出法则 0..6（付费，消耗 5 层铭文）
    22..28  打出法则 0..6（0 消耗，来自昭示）
    29..35  昭示发现：选取法则 0..6
同一种星象/法则的多张副本等价，因此动作按「种类」而不是手牌位置编号，消除了位置排列带来的无关差异。

观测（OBS_DIM = 94，均已缩放到约 0~1）：
    [0:15)    手牌各星象张数 /4
    [15:30)   星轨各星象张数 /3
    [30:33)   星轨已放张数 one-hot（0/1/2）
    [33:40)   付费法则张数 /2          [40:47)  0 消耗法则张数 /2
    [47:62)   抽牌堆各星象张数 /3      [62:77)  弃牌堆各星象张数 /3
    [77]      剩余铭文 /50
    [78:84)   回合 one-hot（1..6）
    [84]      触腕伤害 B × 2
    [85:92)   昭示发现项 one-hot
    [92]      手牌总数 /12             [93]     法则张数 /12
奖励：每一步结算到的倍率增量（×ATK）；训练时再除以 REW_SCALE。回合/对局结束后自动重置并返回新对局的观测。

注意：并行模式下各线程使用各自的随机数状态，不能逐步复现；需要可复现时用 parallel=False。
"""
import numpy as np
from numba import njit, prange

from tulu.env.core import *

NUM_ACTIONS = 36
OBS_DIM = 94
A_STAR0, A_LAW0, A_FREE0, A_OPT0 = 0, 15, 22, 29
MAXSTEPS = 800


@njit(cache=True)
def build_obs_mask(si, sf, cfg, obs, mask):
    for k in range(OBS_DIM):
        obs[k] = 0.0
    for k in range(NUM_ACTIONS):
        mask[k] = False
    for i in range(si[O_HN]):
        obs[si[O_HC + i]] += 0.25
    for i in range(si[O_TN]):
        obs[15 + si[O_TC + i]] += 1.0 / 3.0
    tn = si[O_TN]
    if tn > 2:
        tn = 2
    obs[30 + tn] = 1.0
    for li in range(si[O_LN]):
        lid = si[O_LI + li]
        if si[O_LC + li] > 0:
            obs[33 + lid] += 0.5
        else:
            obs[40 + lid] += 0.5
    for c in range(NT):
        obs[47 + c] = si[O_DP + c] / 3.0
        obs[62 + c] = si[O_DC + c] / 3.0
    obs[77] = si[O_I] / 50.0
    t = si[O_T]
    if t >= 1 and t <= 6:
        obs[78 + t - 1] = 1.0
    obs[84] = sf[F_B] * 2.0
    for j in range(si[O_PN]):
        obs[85 + si[O_PO + j]] = 1.0
    obs[92] = (si[O_HN] + si[O_LN]) / 12.0
    obs[93] = si[O_LN] / 12.0
    # 合法动作
    if si[O_PN] > 0:
        for j in range(si[O_PN]):
            mask[A_OPT0 + si[O_PO + j]] = True
    else:
        for i in range(si[O_HN]):
            mask[si[O_HC + i]] = True
        for li in range(si[O_LN]):
            lid = si[O_LI + li]
            cost = si[O_LC + li]
            if si[O_I] >= cost:
                if cost > 0:
                    mask[A_LAW0 + lid] = True
                else:
                    mask[A_FREE0 + lid] = True


@njit(cache=True)
def apply_primitive(si, sf, cfg, a, st):
    """执行一个原子动作；非法动作返回 False（不应发生，掩码已排除）。"""
    if a < A_LAW0:
        for i in range(si[O_HN]):
            if si[O_HC + i] == a:
                place_star(si, sf, cfg, i, st)
                return True
        return False
    elif a < A_OPT0:
        k = a - A_LAW0
        lid = k % 7
        paid = k < 7
        for li in range(si[O_LN]):
            if si[O_LI + li] == lid and ((si[O_LC + li] > 0) == paid) and si[O_I] >= si[O_LC + li]:
                play_law(si, sf, cfg, li, st)
                return True
        return False
    else:
        lid = a - A_OPT0
        if si[O_PN] > 0:
            for j in range(si[O_PN]):
                if si[O_PO + j] == lid:
                    pick_option(si, cfg, j)
                    return True
        return False


@njit(cache=True)
def vec_reset(SI, SF, ST, STEPS, cfg, obs, mask):
    for e in range(SI.shape[0]):
        init_game(SI[e], SF[e], cfg)
        for k in range(ST.shape[1]):
            ST[e, k] = 0.0
        STEPS[e] = 0
        build_obs_mask(SI[e], SF[e], cfg, obs[e], mask[e])


@njit(cache=True)
def step_one(e, SI, SF, ST, STEPS, cfg, actions, obs, mask, rew, done, fin, bad):
    si = SI[e]
    sf = SF[e]
    st = ST[e]
    before = sf[F_R]
    ok = apply_primitive(si, sf, cfg, actions[e], st)
    STEPS[e] += 1
    rew[e] = sf[F_R] - before
    d = si[O_DONE] == 1
    if not ok:
        bad[e] += 1
        d = True
    if STEPS[e] >= MAXSTEPS:
        d = True
    done[e] = d
    if d:
        st[ST_TOTAL] = sf[F_R]
        st[ST_BFIN] = sf[F_B]
        st[ST_ILEFT] = si[O_I]
        for k in range(ST.shape[1]):
            fin[e, k] = st[k]
            st[k] = 0.0
        init_game(si, sf, cfg)
        STEPS[e] = 0
    build_obs_mask(si, sf, cfg, obs[e], mask[e])


@njit(parallel=True, cache=True)
def vec_step(SI, SF, ST, STEPS, cfg, actions, obs, mask, rew, done, fin, bad):
    for e in prange(SI.shape[0]):
        step_one(e, SI, SF, ST, STEPS, cfg, actions, obs, mask, rew, done, fin, bad)


@njit(cache=True)
def vec_step_serial(SI, SF, ST, STEPS, cfg, actions, obs, mask, rew, done, fin, bad):
    for e in range(SI.shape[0]):
        step_one(e, SI, SF, ST, STEPS, cfg, actions, obs, mask, rew, done, fin, bad)


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


class VecEnv:
    """N 个并行对局。step 返回 (obs, mask, reward, done, finished_stats)；
    finished_stats 为本步刚结束的对局的统计矩阵（行 = 结束的对局，列同 ST_*）。"""

    def __init__(self, cfg, n_envs, seed=0, parallel=True):
        self.cfg = np.asarray(cfg, dtype=np.float64)
        self.n = n_envs
        self.parallel = parallel
        self.seed = seed
        self.SI = np.zeros((n_envs, SI_LEN), dtype=np.int64)
        self.SF = np.zeros((n_envs, SF_LEN), dtype=np.float64)
        self.ST = np.zeros((n_envs, ST_LEN), dtype=np.float64)
        self.STEPS = np.zeros(n_envs, dtype=np.int64)
        self.obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32)
        self.mask = np.zeros((n_envs, NUM_ACTIONS), dtype=np.bool_)
        self.rew = np.zeros(n_envs, dtype=np.float64)
        self.done = np.zeros(n_envs, dtype=np.bool_)
        self.fin = np.zeros((n_envs, ST_LEN), dtype=np.float64)
        self.bad = np.zeros(n_envs, dtype=np.int64)

    def reset(self):
        seed_rng(self.seed)
        vec_reset(self.SI, self.SF, self.ST, self.STEPS, self.cfg, self.obs, self.mask)
        return self.obs.copy(), self.mask.copy()

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        fn = vec_step if self.parallel else vec_step_serial
        fn(self.SI, self.SF, self.ST, self.STEPS, self.cfg, actions,
           self.obs, self.mask, self.rew, self.done, self.fin, self.bad)
        return (self.obs.copy(), self.mask.copy(), self.rew.copy(), self.done.copy(),
                self.fin[self.done].copy())
