"""把滚动前瞻专家蒸馏进神经网络（需要 PyTorch）。

  1. 行为克隆（BC）：在专家自己走过的状态上，最小化交叉熵（策略头）与回报回归损失（价值头）。
  2. DAgger：让学生策略自己打，在它走到的状态上向专家查询标签，并入数据集再训练，缓解分布偏移。
  3. 之后可用 PPO 微调：python scripts/b2_train_ppo.py --init-from <检查点>。
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tulu.env.core import ST_TOTAL, F_R
from tulu.rl.expert import label_batch, MAXS, RET_SCALE
from tulu.rl.networks import ActorCritic, NEG_INF, pick_device
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS


# ---------------------------------------------------------------- 数据集
def save_dataset(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_datasets(paths):
    parts = [dict(np.load(p)) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------- 行为克隆
def train_bc(net, data, device, epochs=20, batch=4096, lr=1e-3, lr_final_frac=0.05, vf_coef=0.5,
             val_frac=0.05, seed=0, log=print):
    """监督训练。返回验证集的 top-1 一致率。"""
    rng = np.random.default_rng(seed)
    n = len(data["act"])
    idx = rng.permutation(n)
    nv = int(n * val_frac)
    va, tr = idx[:nv], idx[nv:]
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"])
    act = torch.from_numpy(data["act"])
    ret = torch.from_numpy(data["ret"])

    def get(ix):
        ix = torch.from_numpy(ix)
        return (obs[ix].float().to(device), mask[ix].to(device), act[ix].to(device), ret[ix].to(device))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps_total = epochs * int(np.ceil(len(tr) / batch))
    step = 0
    acc = float("nan")
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        tot_ce = tot_v = 0.0
        nb = 0
        for s in range(0, len(perm), batch):
            o, m, a, r = get(perm[s:s + batch])
            for g in opt.param_groups:
                g["lr"] = lr * (1 + (lr_final_frac - 1) * step / max(steps_total - 1, 1))
            logits, v = net(o, m)
            ce = F.cross_entropy(logits, a)
            vl = F.mse_loss(v, r)
            loss = ce + vf_coef * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_ce += ce.item()
            tot_v += vl.item()
            nb += 1
            step += 1
        net.eval()
        with torch.no_grad():
            correct = 0
            vce = 0.0
            for s in range(0, len(va), 16384):
                o, m, a, r = get(va[s:s + 16384])
                logits, _ = net(o, m)
                correct += (logits.argmax(-1) == a).sum().item()
                vce += F.cross_entropy(logits, a, reduction="sum").item()
        acc = correct / max(len(va), 1)
        log(f"  [BC] epoch {ep + 1:2d}/{epochs}  train CE={tot_ce / nb:.4f} 价值MSE={tot_v / nb:.4f}  "
            f"验证 CE={vce / max(len(va), 1):.4f}  与专家一致率={acc:.4f}")
    return acc


# ---------------------------------------------------------------- DAgger
def dagger_collect(net, env_cfg, theta, n_games, n_envs=1024, K=12, R=100, beta=0.0, seed=0,
                   device="cpu", log=print):
    """学生策略（以概率 beta 改用专家动作）打 n_games 局，在其访问到的每个状态上向专家取标签。
    返回 (数据集, 学生贪心策略的每局总倍率)。"""
    theta = np.asarray(theta, dtype=np.float64)
    env = VecEnv(env_cfg, n_envs, seed=seed)
    obs, mask = env.reset()
    rng = np.random.default_rng(seed)
    ar = np.arange(n_envs)
    b_obs = np.zeros((n_envs, MAXS, OBS_DIM), dtype=np.float16)
    b_mask = np.zeros((n_envs, MAXS, NUM_ACTIONS), dtype=np.bool_)
    b_act = np.zeros((n_envs, MAXS), dtype=np.int64)
    b_r = np.zeros((n_envs, MAXS))
    cnt = np.zeros(n_envs, dtype=np.int64)
    O, M, A, Rt, T = [], [], [], [], []
    finished = 0
    step = 0
    t0 = time.time()
    labels = np.zeros(n_envs, dtype=np.int64)
    net.eval()
    while finished < n_games:
        step += 1
        r_before = env.SF[:, F_R].copy()
        label_batch(env.SI, env.SF, env.cfg, theta, K, R, seed * 100_000_000 + step * n_envs, labels)
        with torch.no_grad():
            a_stu, _, _ = net.act(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device), greedy=True)
        a_stu = a_stu.cpu().numpy()
        use_exp = rng.random(n_envs) < beta
        act = np.where(use_exp & (labels >= 0), labels, a_stu)
        ok = labels >= 0
        c = cnt[ok]
        e_ok = ar[ok]
        keep = c < MAXS
        e_k, c_k = e_ok[keep], c[keep]
        b_obs[e_k, c_k] = obs[e_k].astype(np.float16)
        b_mask[e_k, c_k] = mask[e_k]
        b_act[e_k, c_k] = labels[e_k]
        b_r[e_k, c_k] = r_before[e_k]
        cnt[e_k] += 1
        obs, mask, rew, done, fin = env.step(act)
        for e in np.nonzero(done)[0]:
            m = int(min(cnt[e], MAXS))
            total = env.fin[e, ST_TOTAL]
            O.append(b_obs[e, :m].copy())
            M.append(b_mask[e, :m].copy())
            A.append(b_act[e, :m].copy())
            Rt.append(((total - b_r[e, :m]) / RET_SCALE).astype(np.float32))
            T.append(total)
            cnt[e] = 0
            finished += 1
        if step % 20 == 0:
            log(f"  [DAgger] step {step} 完成对局 {finished}/{n_games} 学生平均 {np.mean(T) if T else float('nan'):.1f} "
                f"({time.time() - t0:.0f}s)")
    data = {"obs": np.concatenate(O), "mask": np.concatenate(M), "act": np.concatenate(A),
            "ret": np.concatenate(Rt), "totals": np.array(T)}
    return data, np.array(T)
