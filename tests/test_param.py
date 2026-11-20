"""方法 A 的快速检查：策略能跑完整局，结果量级合理，余波/白嫖统计有记录。"""
import numpy as np

from tulu.config import load_cfg
from tulu.env.core import ST_TOTAL, ST_NLAW
from tulu.param.policy import eval_policy, default_theta, no_law_theta, greedy_theta


def test_policies_run():
    cfg = load_cfg()
    for th, mode in [(default_theta(), 1), (no_law_theta(), 0), (greedy_theta(), 0), (default_theta(), 0)]:
        out = eval_policy(th, cfg, 300, 1234, 12, mode)
        r = out[:, ST_TOTAL]
        assert np.isfinite(r).all() and r.min() > 100 and r.max() < 5000
    out = eval_policy(no_law_theta(), cfg, 300, 1234, 12, 0)
    assert out[:, ST_NLAW].sum() == 0
