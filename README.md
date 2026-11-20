# 诞临·图鲁 · 诞临阶段最优出牌研究（PyCharm 项目）

目标：在「诞临」阶段（铭文攒满、6 个出牌回合）里，求星象与铭文法则的**最优出牌策略**，并分析常规情况下的收益。
所有收益都以「×ATK」的倍率记录；退场的「忘却前夜」+80.0 在汇总时加上，不影响策略。

> **当前状态（2026-11-20）**：两条路线都已训练完成。当前最佳单网络 **1295.8 ± 0.3**（含忘却前夜 +80，10 万局评估），
> 方法 A 参数化最优 1229.9，滚动前瞻专家约 1288~1297。尚未突破 1300（差约 4）；「网络引导的滚动前瞻搜索」试点显示可再高约 +10，正在尝试把它蒸馏回网络。
> 进度细节见 [`docs/PROGRESS.md`](docs/PROGRESS.md)，失败与原因分析见 [`docs/FAILURES.md`](docs/FAILURES.md)。
> 旧规则下的结果已作废，仅归档在 `archive/v2_old_rules/`。

## 结果总览（含 +80，评估局数见脚本）

| 策略 | 平均总倍率 |
|---|---|
| 随机合法动作 | 691 |
| 只放牌不打法则 | 873 |
| 贪心 + 法则 | 1156 |
| 方法 B：纯 PPO 从零 | 1196 |
| **方法 A：参数化 + CEM** | **1229.9** |
| 蒸馏：行为克隆 / DAgger | 1267 / 1268 |
| 专家：滚动前瞻 R=200 / R=800 | 约 1288 / 约 1297 |
| PPO 微调 ft1 / ft2 | 1290.4 / 1292.4 |
| **PPO 微调 ft3（当前最佳单网络，`models/ppo_ft3_last_1296.pt`）** | **1295.8 ± 0.3** |
| 四模型集成 | 1296.4（仅 +0.5） |
| 网络 + 搜索（k=3，R=512，改选门槛 4；每局数秒，不是单个网络） | 约 +10，配对差 +9.6 ± 4.7（500 局）；试点阶段 |

做法：先用方法 A 与滚动前瞻专家做**行为克隆 + DAgger 蒸馏**，再对网络做**增强探索的 PPO 微调**。
试过但不如 PPO 与蒸馏的：**DQN 家族**（离线 Q 最佳 1237，在线微调变差）、多候选补牌、集成。原因分析见 [`docs/FAILURES.md`](docs/FAILURES.md)。
模型快照与回退点：`models/`（说明见 `models/README.md`）与 git 标签 `v0.3-baseline-1268`、`v0.4-ft1-1290`、`v0.5-ft2-1293`、`v0.6-ft3-1296`、`v0.7-dqn-failed`。

## 两条路线

| | 方法 A：参数化策略 + CEM | 方法 B：神经网络强化学习 |
|---|---|---|
| 思路 | 一个 15 参数的打分公式（铭文影子价格、各法则额外价值、灾星稀缺惩罚、触腕伤害未来价值）+ 一步前瞻；用交叉熵方法在仿真里搜索参数 | 演员-评论家网络（MLP）+ PPO，在向量化环境里自己打几百万局，学会任意局面下的动作 |
| 优点 | 快、可解释（参数能直接读出含义） | 表达能力强，能学到公式写不出的组合打法 |
| 缺点 | 表达能力受限，得到的是下界 | 慢、更不稳定；需要额外分析才能总结成规则 |
| 代码 | `tulu/param/` | `tulu/rl/` |
| 依赖 | numba、numpy | 另需 PyTorch |

另有两个关键工具：
- **滚动前瞻**（`tulu/param/rollout.py`）——每个决策点对每个候选动作用基础策略模拟打完整局，估计参数化策略离最优还有多远；也用作蒸馏的「专家」（`tulu/rl/expert.py`）。
- **网络引导的滚动前瞻**（`tulu/rl/nnsearch.py`）——以神经网络为基础策略，对网络前 k 个候选动作各模拟 R 次（共同随机数），仅当候选比网络贪心动作好出一个门槛才改选。思路来自专家迭代（Expert Iteration）。

## 目录结构

```
TuluNativity/
├── README.md
├── environment.yml / requirements.txt / pyproject.toml
├── configs/default.json        面板数值 + 规则参数（rho、h 由面板推导）
├── docs/                       model_spec.md 形式化模型与全部规则假设；PROGRESS.md 进度；FAILURES.md 失败与原因分析
│   └──（model_spec.md）          规则假设 A-01 ~ A-32（均已确认，含 2026-11-20 确认的「重衍丢弃的法则触发余波」）
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
│   │   ├── networks.py           演员-评论家网络、决斗式 Q 网络（PyTorch）
│   │   ├── ppo.py                PPO 训练（GAE、动作掩码、检查点、日志）
│   │   ├── evaluate.py           评估网络策略
│   │   ├── expert.py / distill.py  滚动前瞻专家、行为克隆与 DAgger 蒸馏
│   │   ├── nnsearch.py           网络引导的滚动前瞻搜索（专家迭代）
│   │   ├── qexpert.py / qlearn.py  DQN 路线（已验证不如 PPO，留作失败参考）
│   │   └── ensemble.py           多检查点集成
│   └── common/                 统计汇总、中文名称
├── scripts/                    入口脚本（a=方法A，b=方法B，c=对比，d=蒸馏，e=整条流水线，g=集成，h=DQN，i/j/k=网络搜索试点/采集/微调，v=可视化）
├── tests/                      pytest
├── models/                     关键模型快照（体积小，用于回退）
├── results/{param,rl}/         运行结果（脚本自动写入，不进 git）
└── archive/                    v2_old_rules（旧规则，已作废）、failed_dqn（DQN 失败参考的日志与网络）
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
python scripts/e_pipeline.py --tag demo                # 整条流水线：CEM → 专家数据 → 行为克隆 → DAgger → PPO → 对比（规则/面板更新后一条命令重跑）
python scripts/b2_train_ppo.py --name run1             # 单独训练（可 --resume 续训；--init-from 从蒸馏网络出发微调）
python scripts/b3_evaluate_nn.py --ckpt results/rl/run1/best.pt --theta results/param/final/theta.json
python scripts/b4_inspect_policy.py --ckpt results/rl/run1/best.pt             # 看典型局面下的动作偏好

# 网络引导搜索（试点）
python scripts/i_nnsearch_pilot.py --games 200 --k 3 --R 512 --margin 4   # 配对比较纯网络与网络+搜索

# 对比
python scripts/c_compare.py --theta results/param/final/theta.json --ckpt results/rl/run1/best.pt
```

## 配置与最终输入（`configs/default.json`）

| 项 | 取值 |
|---|---|
| 面板（60 级） | 体质 167、攻击 194、防御 108；三者在诞临阶段均再 +30%；每 100 体质 = 289 生命 |
| 造物 | 12 个黄金造物 = 24 个等效白银，加法叠加（攻击 +15%/个，防御 +3%/个） |
| ATK / DEF | 初始攻击 337（已含 +30%；2026-11-20 更新），造物后 ATK ≈ 1550.2、DEF ≈ 322.7（防御/生命与攻击同幅缩放） |
| **ρ = DEF/ATK** | **0.2082**（面板更新后不变） |
| 最大生命 | ≈ 4862.6（潮蚀失去 6 层） |
| **h = H/ATK** | **3.137**（不变，所以已有模型与结果仍然有效；变的只是倍率换算成真实伤害） |
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
- 余波：回合末自动弃置的未使用法则每张 +1 铭文（上限 50）；重衍丢弃的其他未使用法则同样触发余波（2026-11-20 确认）。
- 手牌满 12 时新获得的牌无法进入手牌。

## 已知局限

- 参数化策略只能表达打分公式覆盖的打法，得到的是最优值的下界；滚动前瞻与神经网络用来检验这一点。
- 「最优」指对当前模型假设最优；伤害数值公式预计会有一次大更新，届时需用 `scripts/e_pipeline.py` 重跑。
- 向量化环境在并行模式下各线程的随机数状态独立，不能逐步复现；需要复现时用 `VecEnv(..., parallel=False)`。
- 目前没有在缩小实例上做精确动态规划来验证最优性，列为后续工作。

> **在线试玩**：https://coliner-jiajie.github.io/tulu-nativity/ （GitHub Pages，部署的是 `viz/` 目录；仓库设置里 Pages 的 Source 选「GitHub Actions」）

## 出牌回放可视化（viz/，纯静态网页）

把训练得到的策略（滚动前瞻专家 / 方法 A 参数化最优 / 方法 B PPO）做成可交互的卡牌动画，并可以**亲自上手**出牌、结束后与程序解对比。
整个页面是纯前端：规则引擎与三种策略都已移植成 JavaScript（`viz/engine.js`），**不需要任何服务器**，双击 `viz/index.html` 即可（也可以直接放到 GitHub Pages 等静态托管）。

- **回放模式**：选策略与种子，逐步播放抽牌、星轨结算、法则施放、星潮奔涌；右侧显示每个决策的候选评分。「按种子生成」可对任意种子现场打一局。
- **上手模式**：选种子（同一个种子 = 同一个开局），自己点牌出牌；「专家建议」只给参考不代打。结束后给出同种子单局对比，以及程序在同一开局上重复多局的分布与你的百分位。

文件：

| 文件 | 作用 |
|---|---|
| `viz/index.html` | 页面与动画 |
| `viz/engine.js` | 规则引擎 + 方法 A + 滚动前瞻专家 + PPO 推理（与 `tulu/env/core.py`、`tulu/param/`、`tulu/rl/expert.py` 一一对应） |
| `viz/params.js` | 环境配置、方法 A 的 θ、PPO 权重、真实伤害换算（由 `scripts/v_export_params.py` 生成） |
| `viz/data.js` | 预生成的回放（种子 1~40 × 三策略，由 `scripts/v_gen_data.sh` 生成，约 1 分钟，用 macOS 自带的 jsc） |

规则、配置、策略或模型变了之后，依次重跑：

```bash
python scripts/v_export_params.py      # 更新 viz/params.js
./scripts/v_gen_data.sh                # 更新 viz/data.js
```

JS 引擎与 Python 版的统计一致（2 万局：随机 691.1 vs 691.0；方法 A 1231.0±0.7 vs 1229.96；PPO 1196.5 vs 1196.0）。随机数不同，所以同一个种子在 Python 与 JS 里的开局不同。
`scripts/v_export_episodes.py` 与 `tulu/viz/recorder.py` 是早期的 Python 录制版本，仅作参考，不再被页面使用。
