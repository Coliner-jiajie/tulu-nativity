"""[DQN 路线] 决斗式 Double DQN 系列：用滚动前瞻专家的 Q 值做离线训练，再在线 DQN 微调。需要 PyTorch。
子命令：
  genq    专家打若干局，记录观测/掩码/专家动作/36 维 Q 向量/剩余回报     → results/rl/qdist/<name>.npz
  trainq  离线训练决斗式 Q 网络（回归 Q、优势、间隔损失）并评估         → results/rl/qdist/<out>/last.pt
  dqn     在线 Double DQN 微调（n 步回报、ε-greedy 探索、混入专家数据）     → results/rl/qdist/<out>/best.pt
示例：
  python scripts/h_qdist.py genq --games 2000 --name qpilot
  python scripts/h_qdist.py trainq --data qpilot --out qpilot_net --epochs 40
  python scripts/h_qdist.py dqn --init qpilot_net --demo qpilot --out qpilot_dqn --steps 300"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke, RESULTS_DIR
from tulu.param.cem import load_theta

QD = RESULTS_DIR / "rl" / "qdist"


def _save(path, net, cfg, extra=None):
    import torch
    from tulu.rl.ppo import PPOConfig, save_checkpoint
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    save_checkpoint(path, net, opt, 0, PPOConfig(), cfg, extra or {})


def _eval(net, cfg, nuke, games, device, tag):
    from tulu.rl.evaluate import evaluate_net
    r = evaluate_net(net.to(device), cfg, n_games=games, seed=987654, device=device)
    print(f"[{tag}] 贪心 {games} 局: {r['mean'] + nuke:.2f} ± {r['se']:.2f} "
          f"(p5/50/95 = {r['p05'] + nuke:.0f}/{r['p50'] + nuke:.0f}/{r['p95'] + nuke:.0f})", flush=True)
    return r["mean"] + nuke


def _load(names):
    parts = [dict(np.load(QD / f"{n}.npz")) for n in names]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def cmd_genq(a):
    from tulu.rl.qexpert import generate_q_dataset
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    t0 = time.time()
    data = generate_q_dataset(load_theta(a.theta), cfg, a.games, seed0=a.seed0, K=12, R=a.R, chunk=a.chunk,
                              log=lambda s: print(s, flush=True))
    QD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(QD / f"{a.name}.npz", **data)
    t = data["totals"]
    nc = np.sum(~np.isnan(data["qv"]), axis=1)
    print(f"Q 专家 {len(t)} 局: 平均 {t.mean() + nuke:.2f} ± {t.std() / np.sqrt(len(t)):.2f}；样本 {len(data['act'])} 条；"
          f"含 ≥2 个候选的决策状态 {np.mean(nc >= 2):.1%}；用时 {time.time() - t0:.0f}s", flush=True)


def cmd_trainq(a):
    from tulu.rl.networks import DuelingQNet, pick_device
    from tulu.rl.qlearn import train_q_offline
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    dev = pick_device(a.device)
    data = _load(a.data)
    print(f"数据集 {len(data['act'])} 条", flush=True)
    net = DuelingQNet(hidden=a.hidden, layers=a.layers).to(dev)
    t0 = time.time()
    train_q_offline(net, data, dev, epochs=a.epochs, batch=a.batch, lr=a.lr, w_adv=a.w_adv, w_margin=a.w_margin,
                    log=lambda s: print(s, flush=True))
    _save(QD / a.out / "last.pt", net, cfg)
    print(f"离线训练完成 {time.time() - t0:.0f}s -> {QD / a.out / 'last.pt'}", flush=True)
    _eval(net, cfg, nuke, a.eval_games, dev, "离线 Q")


def cmd_dqn(a):
    import torch
    from tulu.rl.networks import pick_device
    from tulu.rl.ppo import load_net
    from tulu.rl.qlearn import dqn_finetune
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    dev = pick_device(a.device)
    net, ck = load_net(QD / a.init / "last.pt", dev)
    demo = _load(a.demo) if a.demo else None
    out = QD / a.out
    out.mkdir(parents=True, exist_ok=True)
    base = _eval(net, cfg, nuke, a.eval_games, dev, "DQN 起点")
    ev = lambda n: _eval(n, cfg, nuke, a.eval_games, dev, "DQN")
    best_state, best = dqn_finetune(net, cfg, dev, steps=a.steps, n_envs=a.envs, nstep=a.nstep, lr=a.lr,
                                    eps_start=a.eps0, eps_end=a.eps1, eval_every=a.eval_every,
                                    eval_games=a.eval_games, demo=demo, log=lambda s: print(s, flush=True),
                                    evaluate=ev)
    if best_state is not None and best > base:
        net.load_state_dict(best_state)
        _save(out / "best.pt", net, cfg, {"score": best})
        print(f"DQN 微调最佳 {best:.2f}（起点 {base:.2f}）-> {out / 'best.pt'}", flush=True)
    else:
        print(f"DQN 微调未超过起点（最佳 {best:.2f} ≤ 起点 {base:.2f}），不保存", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--theta", default=str(RESULTS_DIR / "param" / "final" / "theta.json"))
    ap.add_argument("--device", default="cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genq")
    g.add_argument("--games", type=int, default=2000)
    g.add_argument("--R", type=int, default=200)
    g.add_argument("--chunk", type=int, default=500)
    g.add_argument("--seed0", type=int, default=3_000_000)
    g.add_argument("--name", default="qdata")
    t = sub.add_parser("trainq")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", default="qnet")
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--w-adv", type=float, default=5.0)
    t.add_argument("--w-margin", type=float, default=1.0)
    t.add_argument("--eval-games", type=int, default=20000)
    d = sub.add_parser("dqn")
    d.add_argument("--init", required=True)
    d.add_argument("--demo", nargs="*", default=None)
    d.add_argument("--out", default="qdqn")
    d.add_argument("--steps", type=int, default=300)
    d.add_argument("--envs", type=int, default=2048)
    d.add_argument("--nstep", type=int, default=8)
    d.add_argument("--lr", type=float, default=3e-5)
    d.add_argument("--eps0", type=float, default=0.05)
    d.add_argument("--eps1", type=float, default=0.01)
    d.add_argument("--eval-every", type=int, default=50)
    d.add_argument("--eval-games", type=int, default=20000)
    a = ap.parse_args()
    {"genq": cmd_genq, "trainq": cmd_trainq, "dqn": cmd_dqn}[a.cmd](a)


if __name__ == "__main__":
    main()
