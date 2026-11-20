"""DQN 类方法（需要 PyTorch）。

阶段 1（离线，DQfD 风格）：用滚动前瞻专家算出的各候选动作的 Q 值，训练决斗式 Q 网络。
    损失 = 候选动作 Q 值回归（Huber）
         + 相对最优候选的优势回归（Huber，权重更大，因为候选之间的差距很小）
         + 大间隔损失：未被专家评估的合法动作的 Q 要比最优候选低一个间隔
         + 价值头回归 V(s) → 最优候选的 Q
阶段 2（在线）：Double DQN + n 步回报 + ε-greedy 探索（可混入专家数据，目标网络软更新），见 dqn_finetune。
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from tulu.env.core import ST_TOTAL
from tulu.rl.networks import DuelingQNet, pick_device
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS

Q_SCALE = 100.0


def train_q_offline(net, data, device, epochs=40, batch=2048, lr=1e-3, lr_final_frac=0.05, w_main=1.0,
                    w_adv=5.0, w_margin=1.0, w_v=0.5, margin_dec=0.05, margin_fol=0.02, val_frac=0.05,
                    seed=0, log=print):
    rng = np.random.default_rng(seed)
    n = len(data["act"])
    idx = rng.permutation(n)
    nv = int(n * val_frac)
    va, tr = idx[:nv], idx[nv:]
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"])
    qv = torch.from_numpy(data["qv"])
    act = torch.from_numpy(data["act"])

    def get(ix):
        ix = torch.from_numpy(ix)
        return obs[ix].float().to(device), mask[ix].to(device), qv[ix].to(device), act[ix].to(device)

    def losses(o, m, qt, a):
        q, v = net.q_values(o, m)
        tgt = qt / Q_SCALE
        has = ~torch.isnan(tgt)
        tmax, astar = torch.where(has, tgt, torch.full_like(tgt, -1e9)).max(1)
        qa = q.gather(1, astar[:, None])
        main = F.huber_loss(q[has], tgt[has], delta=0.5)
        adv_pred = q - qa
        adv_tgt = tgt - tmax[:, None]
        adv = F.huber_loss(adv_pred[has], adv_tgt[has], delta=0.5)
        ncand = has.sum(1)
        mrg = torch.where(ncand >= 2, torch.full_like(tmax, margin_dec), torch.full_like(tmax, margin_fol))[:, None]
        nonc = m & ~has
        viol = F.relu(q - qa.detach() + mrg)
        marg = (viol * nonc).sum() / nonc.sum().clamp(min=1)
        vl = F.huber_loss(v, tmax, delta=0.5)
        total = w_main * main + w_adv * adv + w_margin * marg + w_v * vl
        return total, main, adv, marg, vl, q, has, astar

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps_total = epochs * int(np.ceil(len(tr) / batch))
    step = 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        acc = np.zeros(4)
        nb = 0
        for s in range(0, len(perm), batch):
            o, m, qt, a = get(perm[s:s + batch])
            for g in opt.param_groups:
                g["lr"] = lr * (1 + (lr_final_frac - 1) * step / max(steps_total - 1, 1))
            total, main, adv, marg, vl, *_ = losses(o, m, qt, a)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            acc += [main.item(), adv.item(), marg.item(), vl.item()]
            nb += 1
            step += 1
        net.eval()
        with torch.no_grad():
            correct = 0
            regret = 0.0
            cnt = 0
            vloss = 0.0
            for s in range(0, len(va), 16384):
                o, m, qt, a = get(va[s:s + 16384])
                total, main, adv, marg, vl, q, has, astar = losses(o, m, qt, a)
                pick = q.argmax(1)
                correct += (pick == a).sum().item()
                tgt = qt / Q_SCALE
                tmax = torch.where(has, tgt, torch.full_like(tgt, -1e9)).max(1).values
                chosen = tgt.gather(1, pick[:, None]).squeeze(1)
                ok = ~torch.isnan(chosen)
                regret += (tmax[ok] - chosen[ok]).sum().item()
                cnt += ok.sum().item()
                vloss += vl.item() * len(a)
        acc /= max(nb, 1)
        log(f"  [Q离线] epoch {ep + 1:2d}/{epochs} 回归={acc[0]:.4f} 优势={acc[1]:.4f} 间隔={acc[2]:.4f} 价值={acc[3]:.4f} | "
            f"验证: 与专家一致率={correct / max(len(va), 1):.4f}  已评估动作的遗憾(×100)={100 * regret / max(cnt, 1):.2f}")


def dqn_finetune(net, env_cfg, device, steps=300, n_envs=2048, nstep=8, gamma=1.0, lr=3e-5, batch=4096,
                 updates_per_step=2, buffer_size=1_500_000, eps_start=0.05, eps_end=0.01, tau=0.01,
                 eval_every=50, eval_games=20000, save_path=None, demo=None, demo_frac=0.25, seed=0,
                 log=print, evaluate=None):
    """在线 Double DQN 微调。每一步让 n_envs 个并行环境各走一步（ε-greedy：以概率 ε 走随机合法动作），
    把 n 步回报转移存入回放池，再做若干次小批量更新；目标网络软更新。demo 为专家数据（含 obs/mask/act/qv/ret）时，
    每批混入 demo_frac 的专家样本（用 ret 作为蒙特卡洛目标：Q(s,a_专家) ≈ 剩余回报）。
    返回学习过程中评估最佳的网络状态字典。"""
    rng = np.random.default_rng(seed)
    env = VecEnv(env_cfg, n_envs, seed=seed)
    obs, mask = env.reset()
    tgt_net = DuelingQNet(**{k: v for k, v in net.meta.items() if k in ("obs_dim", "n_actions", "hidden", "layers")}).to(device)
    tgt_net.load_state_dict(net.state_dict())
    tgt_net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    cap = buffer_size
    B_obs = np.zeros((cap, OBS_DIM), dtype=np.float16)
    B_mask = np.zeros((cap, NUM_ACTIONS), dtype=np.bool_)
    B_act = np.zeros(cap, dtype=np.int64)
    B_ret = np.zeros(cap, dtype=np.float32)      # n 步累计奖励（/100）
    B_nobs = np.zeros((cap, OBS_DIM), dtype=np.float16)
    B_nmask = np.zeros((cap, NUM_ACTIONS), dtype=np.bool_)
    B_disc = np.zeros(cap, dtype=np.float32)     # gamma^n，遇到终止则为 0
    pos = 0
    size = 0
    # 每个环境的 n 步滑动窗口
    W_obs = np.zeros((n_envs, nstep, OBS_DIM), dtype=np.float16)
    W_mask = np.zeros((n_envs, nstep, NUM_ACTIONS), dtype=np.bool_)
    W_act = np.zeros((n_envs, nstep), dtype=np.int64)
    W_rew = np.zeros((n_envs, nstep), dtype=np.float32)
    W_len = np.zeros(n_envs, dtype=np.int64)
    ar = np.arange(n_envs)
    best_state, best_score = None, -1e18
    t0 = time.time()

    def push(e_idx, k_idx, nobs_rows, nmask_rows, disc_vals):
        nonlocal pos, size
        m = len(e_idx)
        if m == 0:
            return
        ix = (pos + np.arange(m)) % cap
        B_obs[ix] = W_obs[e_idx, k_idx]
        B_mask[ix] = W_mask[e_idx, k_idx]
        B_act[ix] = W_act[e_idx, k_idx]
        # n 步奖励：从 k_idx 起到窗口末尾
        L = W_len[e_idx]
        rets = np.zeros(m, dtype=np.float32)
        for j in range(m):
            e, k = e_idx[j], k_idx[j]
            seg = W_rew[e, k:L[j]]
            rets[j] = np.sum(seg * (gamma ** np.arange(len(seg))))
        B_ret[ix] = rets
        B_nobs[ix] = nobs_rows
        B_nmask[ix] = nmask_rows
        B_disc[ix] = disc_vals
        pos = (pos + m) % cap
        size = min(size + m, cap)

    for step in range(1, steps + 1):
        eps = eps_start + (eps_end - eps_start) * (step - 1) / max(steps - 1, 1)
        with torch.no_grad():
            q, _ = net.q_values(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
        a = q.argmax(1).cpu().numpy()
        explore = rng.random(n_envs) < eps
        for e in np.nonzero(explore)[0]:
            a[e] = rng.choice(np.nonzero(mask[e])[0])
        # 写入窗口
        k = W_len.copy()
        full = k >= nstep
        if full.any():
            # 窗口已满：先把最旧的转移推入回放池（下一状态是当前 obs），再左移
            e_f = ar[full]
            push(e_f, np.zeros(len(e_f), dtype=np.int64), obs[e_f].astype(np.float16), mask[e_f],
                 np.full(len(e_f), gamma ** nstep, dtype=np.float32))
            W_obs[e_f, :-1] = W_obs[e_f, 1:]
            W_mask[e_f, :-1] = W_mask[e_f, 1:]
            W_act[e_f, :-1] = W_act[e_f, 1:]
            W_rew[e_f, :-1] = W_rew[e_f, 1:]
            W_len[e_f] -= 1
            k = W_len.copy()
        W_obs[ar, k] = obs.astype(np.float16)
        W_mask[ar, k] = mask
        W_act[ar, k] = a
        obs, mask, rew, done, fin = env.step(a)
        W_rew[ar, k] = (rew / Q_SCALE).astype(np.float32)
        W_len[ar] = k + 1
        for e in np.nonzero(done)[0]:
            L = int(W_len[e])
            ks = np.arange(L)
            es = np.full(L, e)
            push(es, ks, np.zeros((L, OBS_DIM), dtype=np.float16), np.zeros((L, NUM_ACTIONS), dtype=np.bool_),
                 np.zeros(L, dtype=np.float32))
            W_len[e] = 0
        if size < batch * 4:
            continue
        # 更新
        net.train()
        for _ in range(updates_per_step):
            nb = int(batch * (1 - (demo_frac if demo is not None else 0)))
            ix = rng.integers(0, size, nb)
            o = torch.from_numpy(B_obs[ix].astype(np.float32)).to(device)
            m = torch.from_numpy(B_mask[ix]).to(device)
            ac = torch.from_numpy(B_act[ix]).to(device)
            r = torch.from_numpy(B_ret[ix]).to(device)
            no = torch.from_numpy(B_nobs[ix].astype(np.float32)).to(device)
            nm = torch.from_numpy(B_nmask[ix]).to(device)
            dc = torch.from_numpy(B_disc[ix]).to(device)
            qpred = net.q_values(o, m)[0].gather(1, ac[:, None]).squeeze(1)
            with torch.no_grad():
                has_next = dc > 0
                qn_online = net.q_values(no, nm | (~has_next[:, None]))[0]          # 终止转移的下一状态掩码为空，避免全 -inf
                a_star = qn_online.argmax(1)
                qn = tgt_net.q_values(no, nm | (~has_next[:, None]))[0].gather(1, a_star[:, None]).squeeze(1)
                target = r + dc * qn * has_next.float()
            loss = F.huber_loss(qpred, target, delta=0.5)
            if demo is not None:
                nd = batch - nb
                di = rng.integers(0, len(demo["act"]), nd)
                do = torch.from_numpy(demo["obs"][di].astype(np.float32)).to(device)
                dm = torch.from_numpy(demo["mask"][di]).to(device)
                da = torch.from_numpy(demo["act"][di]).to(device)
                dr = torch.from_numpy(demo["ret"][di]).to(device)
                qd = net.q_values(do, dm)[0].gather(1, da[:, None]).squeeze(1)
                loss = loss + 0.5 * F.huber_loss(qd, dr, delta=0.5)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for p, tp in zip(net.parameters(), tgt_net.parameters()):
                    tp.mul_(1 - tau).add_(tau * p)
        if step % 10 == 0:
            log(f"  [DQN] step {step}/{steps} ε={eps:.3f} loss={loss.item():.4f} 池={size} ({time.time() - t0:.0f}s)")
        if evaluate is not None and step % eval_every == 0:
            net.eval()
            score = evaluate(net)
            log(f"  [DQN eval] step {step}: 贪心 {score:.2f}")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if save_path is not None:
                    torch.save(best_state, save_path)
    return best_state, best_score
