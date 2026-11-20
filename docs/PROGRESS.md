# 进度记录（2026-11-19 暂停）

用户决定「训练优化先进行到这里」。以下是暂停时的状态，方便之后接着做。

## 已完成

- 规则环境（`tulu/env/core.py`）、方法 A（参数化 + CEM + 滚动前瞻）、方法 B（向量化环境 + PPO）、蒸馏（`tulu/rl/expert.py`、`distill.py`、`scripts/d_distill.py`）全部实现，pytest 通过。
- 规则假设 A-01 ~ A-32 全部确认（见 `model_spec.md`）；仅「重衍的丢弃不触发余波」是我的解读，待确认。
- 最终输入（`configs/default.json`）：ρ = 0.2082，h = 3.137，铭文 50，余波开启，手牌上限 12，总倍率含忘却前夜 +80。

## 结果（含 +80，20 万局评估）

| 策略 | 平均总倍率 |
|---|---|
| 随机合法动作 | 691.0 |
| 只放牌不打法则 | 872.7 |
| 贪心 + 法则 | 1155.6 |
| 方法 B：纯 PPO 神经网络（2000 轮，约 1 小时） | 1196.0 |
| **方法 A：参数化最优（CEM）** | **1229.9** |
| 专家：滚动前瞻（2000 局） | 约 1291（比方法 A 高约 53~61） |
| 蒸馏试点：行为克隆（7.7 万条样本，仅 450 次更新） | 1077.8（欠训练） |

## 文件位置

| 内容 | 位置 |
|---|---|
| 方法 A 最优参数、对比数据 | `results/param/final/theta.json`、`summary.json` |
| 滚动前瞻检验 | `results/param/a3_headroom.log` |
| PPO 训练日志、曲线、模型 | `results/rl/run1/`（`train_log.csv`、`training_curve.png`、`best.pt`、`last.pt`） |
| 蒸馏专家数据、试点网络 | `results/rl/distill/`（`expert_pilot.npz`、`bc_pilot/last.pt`、`pilot.log`） |
| 旧规则结果（已作废） | `archive/v2_old_rules/` |

## 暂停点与下一步（未执行）

放大蒸馏，预计 1.5~2 小时：
1. `python scripts/d_distill.py gen --games 20000 --name expert_big`（约 27 分钟）
2. `python scripts/d_distill.py bc --data expert_big --out bc_big --epochs 60 --batch 2048`（约 5~10 分钟）
3. `python scripts/d_distill.py dagger --init bc_big --data expert_big --out dag_big --rounds 3 --games 3000`（每轮约 15~25 分钟，先跑 1 轮测速）
4. 可选：`python scripts/b2_train_ppo.py --name ft --init-from results/rl/distill/dag_big/last.pt --lr 1e-4`

也可先放大一半（专家 1 万局，约 14 分钟）再决定。其他备选方向：扩充方法 A 的打分特征、网络更大/训练更久/多种子、在缩小实例上做精确动态规划验证最优性、把策略总结成可读的打法规则表。

## 约定

- 重计算/训练只在用户说「计算」后才开始；规则假设逐条确认。
- 计算用 Bash 在后台跑（PyCharm/VS Code 我只有点击权限），结果写到 `results/`。
