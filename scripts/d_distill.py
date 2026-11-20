"""[蒸馏] 把滚动前瞻专家蒸馏进神经网络。需要 PyTorch。
子命令：
  gen     让专家（滚动前瞻 + 方法 A 基础策略）打若干局并记录          → results/rl/distill/<name>.npz
  bc      行为克隆：在专家数据上训练网络，并评估                       → results/rl/distill/<out>/last.pt
  dagger  DAgger：学生自己打、向专家取标签、并入数据再训练，多轮
示例：
  python scripts/d_distill.py gen --games 2000 --name expert_pilot
  python scripts/d_distill.py bc --data expert_pilot --out bc_pilot
  python scripts/d_distill.py dagger --init bc_pilot --data expert_pilot --out dag_pilot --rounds 2 --games 1000
之后可 PPO 微调：python scripts/b2_train_ppo.py --name ft --init-from results/rl/distill/dag_pilot/last.pt --lr 1e-4"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, final_nuke, RESULTS_DIR
from tulu.param.cem import load_theta

DIST = RESULTS_DIR / "rl" / "distill"


def _net_and_eval(a, cfg, nuke, net, tag):
    from tulu.rl.evaluate import evaluate_net
    from tulu.rl.networks import pick_device
    dev = pick_device(a.device)
    r = evaluate_net(net.to(dev), cfg, n_games=a.eval_games, seed=987654, device=dev)
    print(f"[{tag}] 学生(贪心) {a.eval_games} 局: {r['mean'] + nuke:.2f} ± {r['se']:.2f} "
          f"(p5/50/95 = {r['p05'] + nuke:.0f}/{r['p50'] + nuke:.0f}/{r['p95'] + nuke:.0f})", flush=True)
    return r["mean"] + nuke


def _save(path, net, cfg, hidden, layers, extra):
    import torch
    from tulu.rl.ppo import PPOConfig, save_checkpoint
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    save_checkpoint(path, net, opt, 0, PPOConfig(hidden=hidden, layers=layers), cfg, extra)


def cmd_gen(a):
    from tulu.rl.distill import save_dataset
    from tulu.rl.expert import generate_expert_dataset
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    theta = load_theta(a.theta)
    t0 = time.time()
    data = generate_expert_dataset(theta, cfg, a.games, seed0=a.seed0, K=12, R=a.R, chunk=a.chunk,
                                   log=lambda s: print(s, flush=True), ntri=a.ntri)
    out = DIST / f"{a.name}.npz"
    save_dataset(out, data)
    t = data["totals"]
    print(f"专家 {len(t)} 局: 平均总倍率 {t.mean() + nuke:.2f} ± {t.std() / np.sqrt(len(t)):.2f}；"
          f"样本 {len(data['act'])} 条（每局约 {len(data['act']) / len(t):.1f} 步）；"
          f"用时 {time.time() - t0:.0f}s -> {out}")


def cmd_bc(a):
    from tulu.rl.distill import load_datasets, train_bc
    from tulu.rl.networks import ActorCritic, pick_device
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    data = load_datasets([DIST / f"{n}.npz" for n in a.data])
    print(f"数据集 {len(data['act'])} 条", flush=True)
    dev = pick_device(a.device)
    net = ActorCritic(hidden=a.hidden, layers=a.layers).to(dev)
    t0 = time.time()
    acc = train_bc(net, data, dev, epochs=a.epochs, batch=a.batch, lr=a.lr, log=lambda s: print(s, flush=True))
    out = DIST / a.out
    _save(out / "last.pt", net, cfg, a.hidden, a.layers, {"bc_acc": acc})
    print(f"BC 完成 {time.time() - t0:.0f}s，保存 {out / 'last.pt'}", flush=True)
    _net_and_eval(a, cfg, nuke, net, "BC")


def cmd_dagger(a):
    import torch
    from tulu.rl.distill import load_datasets, train_bc, dagger_collect, save_dataset
    from tulu.rl.networks import pick_device
    from tulu.rl.ppo import load_net
    cfg = load_cfg(a.config)
    nuke = final_nuke(a.config)
    theta = load_theta(a.theta)
    dev = pick_device(a.device)
    net, ck = load_net(DIST / a.init / "last.pt", dev)
    hidden, layers = ck["meta"]["hidden"], ck["meta"]["layers"]
    data = load_datasets([DIST / f"{n}.npz" for n in a.data])
    out = DIST / a.out
    log = lambda s: print(s, flush=True)
    history = []
    for rd in range(1, a.rounds + 1):
        t0 = time.time()
        beta = a.beta0 * (0.5 ** (rd - 1))
        new, stu_totals = dagger_collect(net, cfg, theta, a.games, n_envs=a.envs, K=12, R=a.R, beta=beta,
                                         seed=rd + a.seed_offset, device=dev, log=log, eps=a.eps, ntri=a.ntri)
        save_dataset(out / f"round{rd}_new.npz", new)
        print(f"[DAgger 第{rd}轮] β={beta:.2f} 学生访问状态 {len(new['act'])} 条；"
              f"收集阶段学生平均 {stu_totals.mean() + nuke:.2f}；用时 {time.time() - t0:.0f}s", flush=True)
        data = {k: np.concatenate([data[k], new[k]]) for k in data if k in new}
        net.train()
        train_bc(net, data, dev, epochs=a.epochs, batch=a.batch, lr=a.lr, log=log)
        _save(out / "last.pt", net, cfg, hidden, layers, {"round": rd})
        score = _net_and_eval(a, cfg, nuke, net, f"DAgger 第{rd}轮后")
        history.append({"round": rd, "beta": beta, "data": int(len(data["act"])), "eval": score})
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--theta", default=str(RESULTS_DIR / "param" / "final" / "theta.json"))
    ap.add_argument("--device", default="cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=2000)
    g.add_argument("--R", type=int, default=200)
    g.add_argument("--chunk", type=int, default=500)
    g.add_argument("--ntri", type=int, default=1, help="专家的补牌候选方案数")
    g.add_argument("--seed0", type=int, default=1_000_000)
    g.add_argument("--name", default="expert")
    b = sub.add_parser("bc")
    b.add_argument("--data", nargs="+", required=True)
    b.add_argument("--out", default="bc")
    b.add_argument("--hidden", type=int, default=512)
    b.add_argument("--layers", type=int, default=3)
    b.add_argument("--epochs", type=int, default=20)
    b.add_argument("--batch", type=int, default=4096)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--eval-games", type=int, default=20000)
    d = sub.add_parser("dagger")
    d.add_argument("--init", required=True, help="results/rl/distill/ 下的初始网络目录名（来自 bc）")
    d.add_argument("--data", nargs="+", required=True)
    d.add_argument("--out", default="dagger")
    d.add_argument("--rounds", type=int, default=3)
    d.add_argument("--games", type=int, default=2000)
    d.add_argument("--envs", type=int, default=1024)
    d.add_argument("--R", type=int, default=100)
    d.add_argument("--beta0", type=float, default=0.0)
    d.add_argument("--epochs", type=int, default=8)
    d.add_argument("--batch", type=int, default=4096)
    d.add_argument("--lr", type=float, default=5e-4)
    d.add_argument("--eval-games", type=int, default=20000)
    d.add_argument("--seed-offset", type=int, default=0)
    d.add_argument("--ntri", type=int, default=1, help="专家的补牌候选方案数")
    d.add_argument("--eps", type=float, default=0.0, help="探索：学生每步以此概率改走随机合法动作")
    a = ap.parse_args()
    {"gen": cmd_gen, "bc": cmd_bc, "dagger": cmd_dagger}[a.cmd](a)


if __name__ == "__main__":
    main()
