"""[专家迭代] 用搜索标签微调网络（交叉熵，改选样本加权），并用 10 万局评估对比起点。
示例: python scripts/k_finetune_search.py --data s1 --out s1_ft --w-changed 5 --lr 5e-5 --epochs 3"""
import argparse, sys, time, glob
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulu.config import load_cfg, final_nuke
from tulu.rl.ppo import load_net, save_checkpoint, PPOConfig
from tulu.rl.evaluate import evaluate_net

ap = argparse.ArgumentParser()
ap.add_argument("--net", default="models/ppo_ft3_last_1296.pt")
ap.add_argument("--data", nargs="+", default=["s1"])
ap.add_argument("--out", default="s1_ft")
ap.add_argument("--w-changed", type=float, default=5.0)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--eval-games", type=int, default=100000)
ap.add_argument("--seed", type=int, default=424242)
a = ap.parse_args()
cfg = load_cfg(None); nuke = final_nuke(None)
net, ck = load_net(a.net, "cpu")
files = sorted(f for n in a.data for f in glob.glob(f"results/rl/nnsearch/data_{n}_*.npz"))
D = [dict(np.load(f)) for f in files]
O = np.concatenate([d["obs"] for d in D]).astype(np.float32); M = np.concatenate([d["mask"] for d in D])
A = np.concatenate([d["act"] for d in D]); G = np.concatenate([d["greedy"] for d in D])
ch = A != G
print(f"样本 {len(A)}，改选 {ch.sum()} ({ch.mean():.1%})，来自 {len(files)} 个文件", flush=True)
O, M, A, W = map(torch.from_numpy, (O, M, A, np.where(ch, a.w_changed, 1.0).astype(np.float32)))
base = evaluate_net(net, cfg, n_games=a.eval_games, seed=a.seed)
print(f"起点 {a.eval_games} 局: {base['mean'] + nuke:.2f} ± {base['se']:.2f}", flush=True)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
n = len(A)
for ep in range(a.epochs):
    net.train(); perm = torch.randperm(n); tot = 0.0
    for s in range(0, n, a.batch):
        i = perm[s:s + a.batch]
        logits, _ = net(O[i], M[i])
        loss = (F.cross_entropy(logits, A[i], reduction="none") * W[i]).sum() / W[i].sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        tot += loss.item() * len(i)
    net.eval()
    with torch.no_grad():
        lg, _ = net(O, M); pred = lg.argmax(-1)
    print(f"  epoch {ep + 1}: CE={tot / n:.4f} 与搜索标签一致 {(pred == A).float().mean():.4f}（改选样本 {(pred[ch] == A[ch]).float().mean():.3f}）", flush=True)
r = evaluate_net(net, cfg, n_games=a.eval_games, seed=a.seed)
print(f"微调后 {a.eval_games} 局: {r['mean'] + nuke:.2f} ± {r['se']:.2f}（相对起点 {r['mean'] - base['mean']:+.2f}）", flush=True)
out = Path("results/rl/nnsearch") / a.out / "last.pt"
save_checkpoint(out, net, opt, 0, PPOConfig(), cfg, {"score": r["mean"] + nuke}); print("saved", out)
