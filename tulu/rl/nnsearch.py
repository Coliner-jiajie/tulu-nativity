"""以神经网络为基础策略的滚动前瞻搜索（专家迭代的「网络引导搜索」一半）。

在每个有多个合法动作的决策点：取网络概率最高的 k 个候选动作，对每个候选克隆 R 份状态，执行该动作后用网络的
贪心策略打完整局，比较平均总倍率。同一个 r 号模拟在所有候选之间使用相同的随机数序列（共同随机数，按「模拟内
步数」播种），因此候选之间的比较噪声更小。网络贪心动作本身也是候选之一；只有当某个候选比贪心动作好出
margin 以上时才改选（抑制「挑最大值」带来的虚高）。
"""
import time

import numpy as np
import torch
from numba import njit, prange

from tulu.env.core import *
from tulu.rl.vec_env import build_obs_mask, apply_primitive, OBS_DIM, NUM_ACTIONS, MAXSTEPS


@njit(parallel=True, cache=True)
def reset_seeded(SI, SF, cfg, seeds, obs, mask):
    for e in prange(SI.shape[0]):
        np.random.seed(seeds[e])
        init_game(SI[e], SF[e], cfg)
        build_obs_mask(SI[e], SF[e], cfg, obs[e], mask[e])


@njit(parallel=True, cache=True)
def step_seeded(SI, SF, cfg, actions, seeds, active, obs, mask):
    """对 active 的环境执行动作（先按 seeds[e] 播种）。对局结束或动作非法则 active 置为 False。"""
    for e in prange(SI.shape[0]):
        if not active[e]:
            continue
        np.random.seed(seeds[e])
        st = np.zeros(ST_LEN)
        ok = apply_primitive(SI[e], SF[e], cfg, actions[e], st)
        if (not ok) or SI[e, O_DONE] == 1:
            active[e] = False
        else:
            build_obs_mask(SI[e], SF[e], cfg, obs[e], mask[e])


@njit(parallel=True, cache=True)
def build_all(SI, SF, cfg, obs, mask):
    for e in prange(SI.shape[0]):
        build_obs_mask(SI[e], SF[e], cfg, obs[e], mask[e])


@torch.no_grad()
def _greedy(net, obs, mask, device):
    o = torch.from_numpy(obs).to(device)
    m = torch.from_numpy(mask).to(device)
    logits, _ = net(o, m)
    return logits.argmax(-1).cpu().numpy()


@torch.no_grad()
def search_actions(net, SI, SF, cfg, obs, mask, root_seed, k=4, R=32, margin=0.0, device="cpu"):
    """对 E 个根状态返回搜索后的动作（以及网络贪心动作、各候选值，便于诊断）。"""
    E = SI.shape[0]
    o = torch.from_numpy(obs).to(device)
    m = torch.from_numpy(mask).to(device)
    logits, _ = net(o, m)
    kk = min(k, NUM_ACTIONS)
    top = torch.topk(logits, kk, dim=-1)
    cand = top.indices.cpu().numpy()                     # (E, kk)
    cvalid = (top.values > -1e8).cpu().numpy()           # 合法候选
    greedy = cand[:, 0].copy()
    nlegal = mask.sum(1)
    sidx = np.nonzero(nlegal >= 2)[0]                    # 只对有选择的状态搜索
    out = greedy.copy()
    if len(sidx) == 0:
        return out, greedy, None
    # 组装克隆：(根, 候选, r)
    pairs = [(e, j) for e in sidx for j in range(kk) if cvalid[e, j]]
    pe = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    P = len(pairs)
    SIc = np.repeat(SI[pe], R, axis=0)
    SFc = np.repeat(SF[pe], R, axis=0)
    acts0 = np.repeat(cand[pe, pj], R)
    rr = np.tile(np.arange(R), P)
    active = np.ones(P * R, dtype=np.bool_)
    cobs = np.zeros((P * R, OBS_DIM), dtype=np.float32)
    cmask = np.zeros((P * R, NUM_ACTIONS), dtype=np.bool_)
    rs = (root_seed[pe].repeat(R) * 1000003 + rr * 7919).astype(np.int64)
    step_seeded(SIc, SFc, cfg, acts0, rs * 131 + 0, active, cobs, cmask)
    for s in range(1, MAXSTEPS):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        a = _greedy(net, cobs[idx], cmask[idx], device)
        full = np.zeros(P * R, dtype=np.int64)
        full[idx] = a
        step_seeded(SIc, SFc, cfg, full, rs * 131 + s, active, cobs, cmask)
    val = SFc[:, F_R].reshape(P, R).mean(1)              # 每个 (根, 候选) 的平均总倍率
    best_v = {}
    greedy_v = {}
    best_a = {}
    for q in range(P):
        e, j = pe[q], pj[q]
        v = val[q]
        if j == 0:
            greedy_v[e] = v
        if e not in best_v or v > best_v[e]:
            best_v[e] = v
            best_a[e] = cand[e, j]
    for e in sidx:
        if best_v[e] > greedy_v[e] + margin:
            out[e] = best_a[e]
    return out, greedy, (best_v, greedy_v)


def play_games(net, cfg, n_games, seed0, search=True, k=4, R=32, margin=0.0, device="cpu", log=print):
    """n_games 局同时进行，每局按局号播种（搜索版与纯贪心版使用同一批随机数起点，便于配对比较）。
    返回每局总倍率（不含 +80）和被改选的决策比例。"""
    cfg = np.asarray(cfg, dtype=np.float64)
    SI = np.zeros((n_games, SI_LEN), dtype=np.int64)
    SF = np.zeros((n_games, SF_LEN))
    obs = np.zeros((n_games, OBS_DIM), dtype=np.float32)
    mask = np.zeros((n_games, NUM_ACTIONS), dtype=np.bool_)
    gseed = np.arange(n_games, dtype=np.int64) + seed0
    reset_seeded(SI, SF, cfg, gseed * 9973 + 1, obs, mask)
    active = np.ones(n_games, dtype=np.bool_)
    net.eval()
    changed = 0
    nsearched = 0
    t0 = time.time()
    for step in range(MAXSTEPS):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        if search:
            a, g, info = search_actions(net, SI[idx], SF[idx], cfg, obs[idx], mask[idx],
                                        gseed[idx] * 1000 + step, k=k, R=R, margin=margin, device=device)
            if info is not None:
                nsearched += int((mask[idx].sum(1) >= 2).sum())
                changed += int((a != g).sum())
        else:
            a = _greedy(net, obs[idx], mask[idx], device)
        full = np.zeros(n_games, dtype=np.int64)
        full[idx] = a
        step_seeded(SI, SF, cfg, full, gseed * 9973 + 17 + step * 31, active, obs, mask)
        if step % 10 == 0:
            log(f"  [search] step {step} 活跃 {int(active.sum())}/{n_games} ({time.time() - t0:.0f}s)")
    return SF[:, F_R].copy(), changed / max(nsearched, 1)
