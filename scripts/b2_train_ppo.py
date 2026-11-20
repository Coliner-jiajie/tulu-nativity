"""[方法 B] 训练神经网络策略（PPO）。需要 PyTorch。这是真正开始训练的入口，请在你准备好后手动运行。
用法：python scripts/b2_train_ppo.py --name run1 --iterations 2000
续训：python scripts/b2_train_ppo.py --name run1 --resume results/rl/run1/last.pt"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg, RESULTS_DIR
from tulu.rl.ppo import PPOConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="run1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="从检查点（如蒸馏得到的网络）热启动")
    d = PPOConfig()
    for f in fields(PPOConfig):
        v = getattr(d, f.name)
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = PPOConfig(**{f.name: getattr(a, f.name) for f in fields(PPOConfig)})
    train(load_cfg(a.config), cfg, RESULTS_DIR / "rl" / a.name, resume=a.resume, init_from=a.init_from)


if __name__ == "__main__":
    main()
