"""[方法 A] 参数扫描：对某个配置参数取一组值，各自求最优参数化策略。
示例：python scripts/a2_sweep.py --param after --values 0 1 2
      python scripts/a2_sweep.py --param I0 --values 20 30 40 50 60 80
      python scripts/a2_sweep.py --param h --values 0.7 1.8 3.1 4.1"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tulu.config import RESULTS_DIR
from tulu.param.sweep import run_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--param", required=True, help="make_cfg 的参数名，如 rho、h、I0、after、cap")
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--final-games", type=int, default=200000)
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args()
    out = RESULTS_DIR / "param" / f"sweep_{a.param}.json"
    run_sweep(a.param, a.values, out, a.config, a.iters, a.pop, a.games, a.final_games,
              warm_start=not a.no_warm)
    print("saved", out)


if __name__ == "__main__":
    main()
