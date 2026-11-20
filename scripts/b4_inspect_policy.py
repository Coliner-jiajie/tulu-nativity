"""[方法 B] 检查训练好的策略在若干典型局面下的动作偏好，用来归纳「人能读懂的打法规则」。
用法：python scripts/b4_inspect_policy.py --ckpt results/rl/run1/best.pt"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import load_cfg
from tulu.common.names import action_names
from tulu.rl.ppo import load_net
from tulu.rl.networks import pick_device
from tulu.rl.scenario import make_state, observe

SCENARIOS = {
    "手牌无三灾，有重衍（应先白嫖再重衍？）":
        dict(hand=["灾2", "灾3", "垣1", "蕴5"], laws=["重衍", "引辰"], I=50, t=1),
    "三灾同星阶在手（应直接放牌）":
        dict(hand=["灾2", "灾2", "灾2", "蕴5"], laws=["重衍", "引辰"], I=50, t=1),
    "两灾加一张可凑连号的牌，有增辉":
        dict(hand=["灾2", "灾3", "灾5", "垣1"], laws=["增辉", "敛芒"], I=50, t=2),
    "已锁定两张灾星，有易象（锁定后再随机化）":
        dict(hand=["灾3", "垣1", "蕴5"], track=["灾2", "灾3"], laws=["易象"], I=50, t=1),
    "铭文很少时的取舍":
        dict(hand=["灾1", "垣3", "蕴5", "蕴2"], laws=["引辰", "重衍"], I=5, t=4),
    "最后一回合，手牌一般，有引辰":
        dict(hand=["灾4", "灾5", "垣2", "蕴1"], laws=["引辰", "拓途"], I=20, t=6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    dev = pick_device(a.device)
    net, _ = load_net(a.ckpt, dev)
    names = action_names()
    for title, sc in SCENARIOS.items():
        si, sf = make_state(cfg, **sc)
        obs, mask = observe(si, sf, cfg)
        with torch.no_grad():
            logits, v = net(torch.from_numpy(obs[None]).to(dev), torch.from_numpy(mask[None]).to(dev))
        p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        print(f"\n【{title}】 价值估计≈{v.item() * 100:.1f}")
        for i in np.argsort(-p)[:a.top]:
            if mask[i]:
                print(f"   {p[i]:5.1%}  {names[i]}")


if __name__ == "__main__":
    main()
