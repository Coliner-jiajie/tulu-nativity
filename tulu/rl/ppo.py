"""方法 B：PPO（带动作掩码、GAE）。在向量化环境上训练演员-评论家网络（需要 PyTorch）。

训练目标：最大化一局 6 个出牌回合的总倍率（折扣因子 γ=1，有限时域）。
只要调用 train() 才会开始训练；本模块导入时不做任何计算。
"""
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from tulu.env.core import ST_TOTAL
from tulu.rl.networks import ActorCritic, pick_device
from tulu.rl.vec_env import VecEnv, OBS_DIM, NUM_ACTIONS


@dataclass
class PPOConfig:
    n_envs: int = 2048
    rollout_len: int = 64
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    lr_final_frac: float = 0.1       # 学习率线性退火到 lr * lr_final_frac
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rew_scale: float = 100.0         # 奖励除以该值，使价值量级约为 10
    iterations: int = 2000
    hidden: int = 256
    layers: int = 2
    seed: int = 0
    device: str = "auto"
    save_every: int = 50
    eval_every: int = 100
    eval_games: int = 20000


def compute_gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done: (T, N)；last_val: (N,)。done[t]=1 表示第 t 步使对局结束（下一步属于新对局）。"""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros_like(last_val)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        next_v = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def _lin(a, b, frac):
    return a + (b - a) * frac


def collect(env, net, obs, mask, T, device, rew_scale, greedy=False):
    """采集 T 步，返回数据与最后的 (obs, mask)，以及本批结束对局的总倍率列表。"""
    N = env.n
    obs_b = np.zeros((T, N, OBS_DIM), dtype=np.float32)
    mask_b = np.zeros((T, N, NUM_ACTIONS), dtype=np.bool_)
    act_b = np.zeros((T, N), dtype=np.int64)
    logp_b = np.zeros((T, N), dtype=np.float32)
    val_b = np.zeros((T, N), dtype=np.float32)
    rew_b = np.zeros((T, N), dtype=np.float32)
    done_b = np.zeros((T, N), dtype=np.float32)
    totals = []
    for t in range(T):
        obs_b[t] = obs
        mask_b[t] = mask
        o = torch.from_numpy(obs).to(device)
        m = torch.from_numpy(mask).to(device)
        a, logp, v = net.act(o, m, greedy=greedy)
        a_np = a.cpu().numpy()
        act_b[t] = a_np
        logp_b[t] = logp.cpu().numpy()
        val_b[t] = v.cpu().numpy()
        obs, mask, rew, done, fin = env.step(a_np)
        rew_b[t] = rew / rew_scale
        done_b[t] = done
        if len(fin):
            totals.extend(fin[:, ST_TOTAL].tolist())
    o = torch.from_numpy(obs).to(device)
    m = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        last_val = net(o, m)[1].cpu().numpy()
    return (obs_b, mask_b, act_b, logp_b, val_b, rew_b, done_b, last_val), obs, mask, totals


def update(net, opt, batch, cfg, ent_coef, device):
    obs_b, mask_b, act_b, logp_b, val_b, rew_b, done_b, last_val = batch
    adv, ret = compute_gae(rew_b, val_b, done_b, last_val, cfg.gamma, cfg.lam)
    T, N = act_b.shape
    flat = lambda x: x.reshape(T * N, *x.shape[2:])
    obs_t = torch.from_numpy(flat(obs_b)).to(device)
    mask_t = torch.from_numpy(flat(mask_b)).to(device)
    act_t = torch.from_numpy(flat(act_b)).to(device)
    logp_old = torch.from_numpy(flat(logp_b)).to(device)
    adv_t = torch.from_numpy(flat(adv).astype(np.float32)).to(device)
    ret_t = torch.from_numpy(flat(ret).astype(np.float32)).to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = T * N
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
    cnt = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg.minibatch):
            idx = perm[s:s + cfg.minibatch]
            logp, ent, v = net.evaluate_actions(obs_t[idx], mask_t[idx], act_t[idx])
            ratio = torch.exp(logp - logp_old[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[idx]
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * (v - ret_t[idx]).pow(2).mean()
            loss = pi_loss + cfg.vf_coef * v_loss - ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pi"] += pi_loss.item()
                stats["v"] += v_loss.item()
                stats["ent"] += ent.mean().item()
                stats["kl"] += (logp_old[idx] - logp).mean().item()
                stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                cnt += 1
    return {k: v / max(cnt, 1) for k, v in stats.items()}


def save_checkpoint(path, net, opt, it, cfg, env_cfg, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it,
                "ppo_cfg": asdict(cfg), "env_cfg": np.asarray(env_cfg).tolist(),
                "meta": net.meta, "extra": extra or {}}, path)


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic(**ck["meta"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def train(env_cfg, cfg=None, out_dir="results/rl/run", resume=None, init_from=None, log=print):
    """开始训练。每 save_every 轮保存检查点，每 eval_every 轮用贪心策略评估一次并记录最佳模型。"""
    from tulu.rl.evaluate import evaluate_net      # 延迟导入，避免循环依赖

    cfg = cfg or PPOConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    env = VecEnv(env_cfg, cfg.n_envs, seed=cfg.seed)
    obs, mask = env.reset()
    if init_from:
        net, _ = load_net(init_from, device)      # 例如行为克隆 / DAgger 得到的网络
        net.train()
    else:
        net = ActorCritic(hidden=cfg.hidden, layers=cfg.layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    start = 0
    best = -1e18
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["iter"] + 1
        best = ck.get("extra", {}).get("best", best)
    csv_path = out / "train_log.csv"
    new_file = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["iter", "steps", "mean_return", "episodes", "pi_loss", "v_loss", "entropy", "kl",
                    "clipfrac", "lr", "sec", "eval_mean", "eval_se"])
    recent = []
    t_start = time.time()
    for it in range(start, cfg.iterations):
        frac = it / max(cfg.iterations - 1, 1)
        lr = _lin(cfg.lr, cfg.lr * cfg.lr_final_frac, frac)
        for g in opt.param_groups:
            g["lr"] = lr
        ent_coef = _lin(cfg.ent_coef, cfg.ent_coef_final, frac)
        t0 = time.time()
        batch, obs, mask, totals = collect(env, net, obs, mask, cfg.rollout_len, device, cfg.rew_scale)
        st = update(net, opt, batch, cfg, ent_coef, device)
        recent.extend(totals)
        recent = recent[-5000:]
        mean_ret = float(np.mean(recent)) if recent else float("nan")
        ev_mean = ev_se = ""
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            r = evaluate_net(net, env_cfg, n_games=cfg.eval_games, seed=123456, device=device)
            ev_mean, ev_se = r["mean"], r["se"]
            if ev_mean > best:
                best = ev_mean
                save_checkpoint(out / "best.pt", net, opt, it, cfg, env_cfg, {"best": best})
            log(f"[eval] iter {it + 1} greedy mean={ev_mean:.2f} ± {ev_se:.2f} (best {best:.2f})")
        w.writerow([it, (it + 1) * cfg.rollout_len * cfg.n_envs, mean_ret, len(totals), st["pi"], st["v"],
                    st["ent"], st["kl"], st["clipfrac"], lr, time.time() - t0, ev_mean, ev_se])
        f.flush()
        if (it + 1) % 10 == 0:
            log(f"iter {it + 1:5d} return={mean_ret:8.2f} ent={st['ent']:.3f} kl={st['kl']:.4f} "
                f"v={st['v']:.4f} lr={lr:.2e} ({time.time() - t0:.1f}s/iter)")
        if cfg.save_every and (it + 1) % cfg.save_every == 0:
            save_checkpoint(out / "last.pt", net, opt, it, cfg, env_cfg, {"best": best})
    save_checkpoint(out / "last.pt", net, opt, cfg.iterations - 1, cfg, env_cfg, {"best": best})
    f.close()
    log(f"training done in {time.time() - t_start:.0f}s; best greedy eval = {best:.2f}")
    return net
