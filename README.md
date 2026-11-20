# 诞临·图鲁 · 诞临阶段最优出牌研究（PyCharm 项目）

目标：在「诞临」阶段（铭文攒满、6 个出牌回合）里，求星象与铭文法则的**最优出牌策略**，并分析常规情况下的收益。
所有收益都以「×ATK」的倍率记录；退场的「忘却前夜」+80.0 在汇总时加上，不影响策略。

> **当前状态**：项目、代码和测试已搭好；**尚未做任何训练、调参或扫描**。
> 之前在旧规则下得到的扫描结果已作废，仅归档在 `archive/v2_old_rules/`。

## 两条路线

| | 方法 A：参数化策略 + CEM | 方法 B：神经网络强化学习 |
|---|---|---|
| 思路 | 一个 15 参数的打分公式（铭文影子价格、各法则额外价值、灾星稀缺惩罚、触腕伤害未来价值）+ 一步前瞻；用交叉熵方法在仿真里搜索参数 | 演员-评论家网络（MLP）+ PPO，在向量化环境里自己打几百万局，学会任意局面下的动作 |
| 优点 | 快、可解释（参数能直接读出含义） | 表达能力强，能学到公式写不出的组合打法 |
| 缺点 | 表达能力受限，得到的是下界 | 慢、更不稳定；需要额外分析才能总结成规则 |
| 代码 | `tulu/param/` | `tulu/rl/` |
| 依赖 | numba、numpy | 另需 PyTorch |

另有一个检验工具：**滚动前瞻**（`tulu/param/rollout.py`）——每个决策点对每个候选动作用基础策略模拟打完整局，估计参数化策略离最优还有多远。

## 目录结构

```
TuluNativity/
├── README.md
├── environment.yml / requirements.txt / pyproject.toml
├── configs/default.json        面板数值 + 规则参数（rho、h 由面板推导）
├── docs/model_spec.md          形式化模型与全部规则假设 A-01 ~ A-32（均已确认；仅 A-30 中「重衍的丢弃不触发余波」是我的解读，待确认）
├── tulu/
│   ├── config.py               读取配置、由面板推导 ρ=DEF/ATK 与 h=H/ATK
│   ├── env/core.py             ★规则环境（numba），两条路线共用
│   ├── param/                  方法 A
│   │   ├── policy.py             参数化策略（打分、前瞻、锁定/白嫖逻辑）与批量评估
│   │   ├── cem.py                CEM 调参、保存/读取 θ
│   │   ├── rollout.py            滚动前瞻（检验用）
│   │   └── sweep.py              参数扫描
│   ├── rl/                     方法 B
│   │   ├── vec_env.py            向量化环境：36 维动作 + 合法动作掩码 + 94 维观测
│   │   ├── scenario.py           构造指定局面（检查策略用）
│   │   ├── networks.py           演员-评论家网络（PyTorch）
│   │   ├── ppo.py                PPO 训练（GAE、动作掩码、检查点、日志）
│   │   └── evaluate.py           评估网络策略
│   └── common/                 统计汇总、中文名称
├── scripts/                    入口脚本（a=方法A，b=方法B，c=对比）
├── tests/                      pytest
├── results/{param,rl}/         运行结果（脚本自动写入）
└── archive/v2_old_rules/       旧规则下的结果与代码（已作废，仅留档）
```

## 在 PyCharm 中打开

1. 用 Anaconda 创建环境（只做一次）：
   ```bash
   conda env create -f environment.yml
   conda activate tulu
   ```
   （Apple Silicon 上 `pip` 安装的 PyTorch 自带 MPS 加速，训练会自动使用；只做方法 A 可以不装 torch。）
2. PyCharm：`File → Open…` 选择本目录 `TuluNativity`；`Settings → Project → Python Interpreter → Add Interpreter → Conda Environment → Existing`，选 `tulu` 环境。
3. 把 `TuluNativity` 保持为 Sources Root（默认即可）；脚本从项目根目录运行。
4. 首次运行任何脚本会有 numba 编译，大约 10 到 60 秒，之后有缓存。

## 运行顺序（建议）

```bash
python -m pytest -q                      # 检查规则、环境、策略（没装 torch 时自动跳过网络测试）
python scripts/b1_smoke_vecenv.py        # 向量化环境冒烟测试（不是训练）

# 方法 A
python scripts/a1_tune_cem.py --name final            # CEM 求最优参数 + 基线对比
python scripts/a3_rollout_headroom.py --theta results/param/final/theta.json   # 离最优还有多远
python scripts/a2_sweep.py --param after --values 0 1 2                        # 参数扫描示例

# 方法 B（需要 PyTorch）
python scripts/b2_train_ppo.py --name run1             # 训练（可 --resume 续训）
python scripts/b3_evaluate_nn.py --ckpt results/rl/run1/best.pt --theta results/param/final/theta.json
python scripts/b4_inspect_policy.py --ckpt results/rl/run1/best.pt             # 看典型局面下的动作偏好

# 对比
python scripts/c_compare.py --theta results/param/final/theta.json --ckpt results/rl/run1/best.pt
```

## 配置与最终输入（`configs/default.json`）

| 项 | 取值 |
|---|---|
| 面板（60 级） | 体质 167、攻击 194、防御 108；三者在诞临阶段均再 +30%；每 100 体质 = 289 生命 |
| 造物 | 12 个黄金造物 = 24 个等效白银，加法叠加（攻击 +15%/个，防御 +3%/个） |
| ATK / DEF | 194×1.3×(1+0.15×24) ≈ 1160.1 / 108×1.3×(1+0.03×24) ≈ 241.5 |
| **ρ = DEF/ATK** | **0.2082** |
| 最大生命 | 167×1.3×2.89×(1+0.8×6) ≈ 3639（潮蚀失去 6 层） |
| **h = H/ATK** | **3.137** |
| 铭文 | 初始 50，上限 50；每张法则消耗 5；未使用的法则回合末返还 1（余波） |
| 手牌上限 | 12（星象与法则共用） |
| 触腕伤害 | 初值 5%·H，每回合 +1.25%·H，均按攻击力归一；×1.25 的铭文放大只作用于增长与蕴星收益 |

改规则或数值只需改 `configs/default.json`（或脚本参数），不需要改代码。

## 动作与观测（方法 B）

- 36 个动作：0–14 放入某种星象；15–21 打出付费法则；22–28 打出 0 消耗法则（昭示选出）；29–35 昭示发现项选择。同种副本等价，所以按「种类」而不是手牌位置编号。
- 94 维观测：手牌/星轨/抽牌堆/弃牌堆的各星象张数、法则张数、剩余铭文、回合 one-hot、触腕伤害 B、昭示发现项等（详见 `tulu/rl/vec_env.py` 顶部注释）。
- 奖励：每一步结算到的倍率增量；训练时除以 100。γ = 1（有限时域）。

## 规则要点（完整版见 `docs/model_spec.md`）

- 牌库 45 张星象（15 种 × 3 张）；每回合抽 4 张星象 + 随机 2 张法则；星象免费；放满 3 张自动发动星潮奔涌并结束回合。
- **法则可在放第 3 张前任意时刻打出**：重衍前可先把最好的两张放进星轨白嫖卡效；易象/增辉/敛芒前可先锁定好牌。
- 余波：回合末自动弃置的未使用法则每张 +1 铭文（上限 50）；重衍的丢弃不触发。
- 手牌满 12 时新获得的牌无法进入手牌。

## 已知局限

- 参数化策略只能表达打分公式覆盖的打法，得到的是最优值的下界；滚动前瞻与神经网络用来检验这一点。
- 向量化环境在并行模式下各线程的随机数状态独立，不能逐步复现；需要复现时用 `VecEnv(..., parallel=False)`。
- 目前没有在缩小实例上做精确动态规划来验证最优性，列为后续工作。

## 出牌回放可视化（viz/）

把训练得到的策略（滚动前瞻专家 / 方法 A 参数化最优 / 方法 B PPO）打出的整局做成可交互的卡牌动画：
抽牌飞入、星象放入星轨并结算卡效、法则施放与星象变形、星潮奔涌逐次命中、回合末弃牌与余波，右侧同步显示每一步的候选动作与评分。

```bash
python scripts/v_export_episodes.py --games 40      # 录制回放数据 → viz/data.js（约 40 秒，随时可重跑）
open viz/index.html                                  # 直接双击/打开即可，无需服务器、无需联网
```

录制逻辑在 `tulu/viz/recorder.py`（逐步复现各策略的决策并记录候选评分），前端是单文件 `viz/index.html`（原生 JS，无依赖）。
同一个种子在三种策略下是同一个开局，可以切换策略对比。
