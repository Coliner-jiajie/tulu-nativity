"""[流水线] 一条命令按阶段重跑「方法 A → 专家数据 → 行为克隆 → DAgger → PPO 微调 → 对比」。
伤害公式或数值更新后，只需改 configs/*.json（或换 --config），再运行本脚本即可重新得到最优策略。

阶段：cem（方法 A 调参，给专家当基础策略）→ gen（专家数据）→ bc → dagger → ppo（可选）→ compare
示例（完整重跑，约 2~3 小时）：
    python scripts/e_pipeline.py --tag v2 --stages cem gen bc dagger compare
只跑后半段（已有 cem 与专家数据）：
    python scripts/e_pipeline.py --tag v2 --stages bc dagger compare
所有产物都带 --tag 前缀，旧结果不会被覆盖。"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
S = ROOT / "scripts"


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    print(f"  -> exit {r.returncode}，用时 {time.time() - t0:.0f}s", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="本次运行的名字，所有产物以它命名")
    ap.add_argument("--config", default=None, help="配置 JSON，默认 configs/default.json")
    ap.add_argument("--stages", nargs="+", default=["cem", "gen", "bc", "dagger", "compare"],
                    choices=["cem", "gen", "bc", "dagger", "ppo", "compare"])
    ap.add_argument("--gen-parts", type=int, default=4)
    ap.add_argument("--gen-games", type=int, default=5000, help="每份专家数据的局数")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--bc-batch", type=int, default=1024)
    ap.add_argument("--dagger-rounds", type=int, default=3)
    ap.add_argument("--dagger-games", type=int, default=3000)
    ap.add_argument("--ppo-iters", type=int, default=500)
    a = ap.parse_args()
    cfg = ["--config", a.config] if a.config else []
    tag = a.tag
    theta = f"results/param/{tag}/theta.json"
    parts = [f"{tag}_expert_{i}" for i in range(1, a.gen_parts + 1)]

    if "cem" in a.stages:
        run([PY, S / "a1_tune_cem.py", *cfg, "--name", tag, "--iters", 40, "--pop", 96, "--games", 5000])
    if "gen" in a.stages:
        for i, name in enumerate(parts, 1):
            run([PY, S / "d_distill.py", *cfg, "--theta", theta, "gen", "--games", a.gen_games, "--R", a.R,
                 "--seed0", 2_000_000 + i * 100_000, "--name", name])
    if "bc" in a.stages:
        run([PY, S / "d_distill.py", *cfg, "--theta", theta, "bc", "--data", *parts, "--out", f"{tag}_bc",
             "--epochs", a.bc_epochs, "--batch", a.bc_batch])
    if "dagger" in a.stages:
        run([PY, S / "d_distill.py", *cfg, "--theta", theta, "dagger", "--init", f"{tag}_bc", "--data", *parts,
             "--out", f"{tag}_dagger", "--rounds", a.dagger_rounds, "--games", a.dagger_games])
    if "ppo" in a.stages:
        run([PY, S / "b2_train_ppo.py", *cfg, "--name", f"{tag}_ppo", "--init-from",
             f"results/rl/distill/{tag}_dagger/last.pt", "--iterations", a.ppo_iters, "--lr", 1e-4,
             "--device", "cpu", "--eval-every", 100])
    if "compare" in a.stages:
        ck = f"results/rl/distill/{tag}_dagger/last.pt"
        if not (ROOT / ck).exists():
            ck = f"results/rl/distill/{tag}_bc/last.pt"
        run([PY, S / "c_compare.py", *cfg, "--theta", theta, "--ckpt", ck, "--games", 100000])


if __name__ == "__main__":
    main()
