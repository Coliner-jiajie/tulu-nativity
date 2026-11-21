# Tulu · Optimal Play in the Theophany Stage

[中文](README.md) | **English**

Goal: for **Tulu: Godking** in *Morimens* (忘却前夜), find the **optimal card-play policy** (Asterisms and Runes) during the **Theophany** (诞临) stage, assuming the Etched Memory bar is full and the stage lasts 6 turns, and quantify the expected payoff.
All payoffs are recorded as multipliers of ATK ("×ATK"). The +80.0 from the 忘却前夜 (Morimens) skill on retreat is added when reporting totals; it does not affect the policy.

> Terminology follows the official English text (as used in the in-repo web visualization): 诞临 Theophany · 星象 Asterism · 灾/垣/蕴 Antares/Scutum/Pleiades · 星阶 Brilliance · 法则 Rune · 星轨 Orbit · 星潮奔涌 Stars Surge · 铭文 Etched Memory · 余波 Aftershock · 触腕伤害 Tentacle DMG. The seven Runes are 易象 Alter, 增辉 Gleam, 敛芒 Dim, 昭示 Herald, 引辰 Beckon, 拓途 Expand, 重衍 Recast. The long-form documents in `docs/` are in Chinese.

> **Status (2026-11-20)**: both approaches are trained. The best single network is **ft4, 1297.26 ± 0.33** (including the +80, 100k games, current rules). Parametric optimum: 1229.9. Rollout expert: about 1288–1297. A single network has not broken 1300. A network-guided rollout search is about 10 higher (≈1308) but could not be distilled back into one network (see `docs/FAILURES.md`, F11). Optimization was paused on 2026-11-20.
> Progress details: [`docs/PROGRESS.md`](docs/PROGRESS.md). Failures and root-cause analysis: [`docs/FAILURES.md`](docs/FAILURES.md). Results under older rules are obsolete and archived in `archive/v2_old_rules/`.

## Results (including +80)

| Policy | Mean total multiplier |
|---|---|
| Random legal actions | 691 |
| Place Asterisms only, never play Runes | 873 |
| Greedy + Runes | 1156 |
| Approach B: PPO from scratch | 1196 |
| **Approach A: parametric policy + CEM** | **1229.9** |
| Distillation: behavior cloning / DAgger | 1267 / 1268 |
| Expert: rollout lookahead R=200 / R=800 | ≈1288 / ≈1297 |
| PPO fine-tune ft1 / ft2 | 1290.4 / 1292.4 |
| PPO fine-tune ft3 (best under the old rules) | 1295.8 ± 0.3 (1295.67 under current rules) |
| **PPO fine-tune ft4 (current rules; best single network, `models/ppo_ft4_last_1297_newrule.pt`)** | **1297.26 ± 0.33** |
| Ensemble of four checkpoints | 1296.4 (only +0.5) |
| Network + search (ft4 as base policy, k=3, R=512, switch threshold 4; seconds per game, not a single network) | ≈1308 (4000 games; about +10 over the plain network); distilling it back failed, see F11 |

What worked: **behavior cloning + DAgger distillation** from approach A and the rollout expert, followed by **PPO fine-tuning with extra exploration**.
What did not beat PPO plus distillation: the **DQN family** (best offline Q 1237; online fine-tuning got worse), multi-candidate completion, ensembling, and cross-entropy distillation of search labels. Analysis: [`docs/FAILURES.md`](docs/FAILURES.md).
Model snapshots and rollback points: `models/` (see `models/README.md`) and git tags `v0.3-baseline-1268`, `v0.4-ft1-1290`, `v0.5-ft2-1293`, `v0.6-ft3-1296`, `v0.7-dqn-failed`, `v0.8-ft4-1297`.

## Two approaches

| | Approach A: parametric policy + CEM | Approach B: neural-network RL |
|---|---|---|
| Idea | A 15-parameter scoring formula (inscription shadow price, per-Rune extra value, Antares scarcity penalty, future value of Tentacle DMG) with one-step lookahead; parameters tuned by the cross-entropy method in simulation | Actor-critic MLP trained with PPO in a vectorized environment over millions of games; learns an action for any state |
| Pros | Fast and interpretable (parameters have direct meaning) | Expressive; can learn combinations a formula cannot express |
| Cons | Limited expressiveness; yields a lower bound | Slower, less stable; needs extra analysis to extract readable rules |
| Code | `tulu/param/` | `tulu/rl/` |
| Dependencies | numba, numpy | additionally PyTorch |

Two further tools:
- **Rollout lookahead** (`tulu/param/rollout.py`): at each decision point, simulate whole games for every candidate action with a base policy, to estimate how far a policy is from optimal. It is also the "expert" for distillation (`tulu/rl/expert.py`).
- **Network-guided rollout search** (`tulu/rl/nnsearch.py`): use the neural network as the base policy; simulate each of the network's top-k candidate actions R times (common random numbers) and switch away from the network's greedy action only if a candidate beats it by a margin. The idea comes from Expert Iteration.

## Repository layout

```
TuluNativity/
├── README.md / README.en.md
├── environment.yml / requirements.txt / pyproject.toml
├── configs/default.json        panel values + rule parameters (rho and h derived from the panel)
├── docs/                       model_spec.md (formal model and all rule assumptions A-01..A-32),
│                               PROGRESS.md, FAILURES.md, strategy_summary.md, decision_table.md
├── tulu/
│   ├── config.py               load config; derive rho = DEF/ATK and h = H/ATK from the panel
│   ├── env/core.py             * rules environment (numba), shared by both approaches
│   ├── param/                  approach A (policy.py, cem.py, rollout.py, sweep.py)
│   ├── rl/                     approach B
│   │   ├── vec_env.py            vectorized env: 36 actions + legal-action mask + 94-dim observation
│   │   ├── networks.py           actor-critic, dueling Q-network (PyTorch)
│   │   ├── ppo.py                PPO (GAE, action masking, checkpoints, logs)
│   │   ├── expert.py / distill.py  rollout expert; behavior cloning and DAgger
│   │   ├── nnsearch.py           network-guided rollout search
│   │   ├── qexpert.py / qlearn.py  DQN route (verified inferior; kept as a failure reference)
│   │   └── evaluate.py / ensemble.py / scenario.py
│   └── common/                 statistics, display names
├── scripts/                    entry points (a=approach A, b=approach B, c=compare, d=distill,
│                               e=full pipeline, g=ensemble, h=DQN, i/j/k=network search pilot/collect/fine-tune, v=visualization)
├── tests/                      pytest
├── models/                     key model snapshots (small, for rollback)
├── results/{param,rl}/         run outputs (written by scripts, not in git)
├── archive/                    v2_old_rules (obsolete), failed_dqn (logs and networks of the failed DQN attempt)
└── viz/                        static web visualization (replay and play-it-yourself)
```

## Setup

```bash
conda env create -f environment.yml
conda activate tulu
```

PyTorch is only needed for approach B. Use the conda-forge build: the pip build of PyTorch conflicts with numba's OpenMP runtime (OMP Error #15). The first run of any script compiles with numba (10–60 s) and is cached afterwards. Scripts are run from the project root.

## Typical workflow

```bash
python -m pytest -q                      # check rules, environment, policies
python scripts/b1_smoke_vecenv.py        # vectorized-environment smoke test (not training)

# Approach A
python scripts/a1_tune_cem.py --name final
python scripts/a3_rollout_headroom.py --theta results/param/final/theta.json   # how far from optimal
python scripts/a2_sweep.py --param after --values 0 1 2

# Approach B (needs PyTorch)
python scripts/e_pipeline.py --tag demo                # full pipeline: CEM -> expert data -> BC -> DAgger -> PPO -> compare
python scripts/b2_train_ppo.py --name run1             # train alone (--resume; --init-from to fine-tune a distilled net)
python scripts/b3_evaluate_nn.py --ckpt results/rl/run1/best.pt --theta results/param/final/theta.json
python scripts/b4_inspect_policy.py --ckpt results/rl/run1/best.pt

# Network-guided search (pilot)
python scripts/i_nnsearch_pilot.py --games 200 --k 3 --R 512 --margin 4

# Compare
python scripts/c_compare.py --theta results/param/final/theta.json --ckpt results/rl/run1/best.pt
```

## Configuration and final inputs (`configs/default.json`)

| Item | Value |
|---|---|
| Panel | **= (level-60 base + (level − 60) × per-level growth) × 1.3 (灵塑 bonus; English name not confirmed)**. Official level-60 base: CON 167, ATK 194, DEF 108. Current `panel.level` = 90, so ATK = (194 + 30 × 2.17436) × 1.3 = 337. 289 HP per 100 CON. The per-level growth values (ATK 2.17436, DEF 1.21047, CON 1.87174) are back-solved from "level-90 ATK = 337" and assumed proportional, so ρ and h do not depend on level. Replace `growth_per_level` in the config once official values are known. |
| 造物 (artifacts; English name not confirmed) | 12 gold = 24 silver-equivalents, additive (ATK +15% each, DEF +3% each) |
| ATK / DEF | ATK ≈ 1550.2, DEF ≈ 322.7 after the artifacts. To change level: `derive_from_panel(panel, level)` / `display_params(level=…)` / `load_cfg(level=…)` |
| **ρ = DEF/ATK** | **0.2082** |
| Max HP | ≈ 4862.6 (6 layers of tide erosion lost) |
| **h = H/ATK** | **3.137** |
| Etched Memory | start 50, cap 50; each Rune costs 5; each unused Rune refunds 1 at turn end (Aftershock) |
| Hand limit | 12 (Asterisms and Runes share it) |
| Tentacle DMG B | initial 5% of H, +1.25% of H per turn, all normalized by ATK; the ×1.25 inscription amplification applies only to growth and Pleiades gains |

Changing rules or numbers only needs editing `configs/default.json` (or script arguments), not code.

## Actions and observations (approach B)

- 36 actions: 0–14 place an Asterism of a given kind and Brilliance; 15–21 play a paid Rune; 22–28 play a zero-cost Rune (from Herald); 29–35 choose a Herald option. Copies of the same card are equivalent, so actions are indexed by kind rather than by hand position.
- 94-dim observation: counts of each Asterism in hand / Orbit / draw pile / discard pile, Rune counts, remaining Etched Memory, one-hot turn, Tentacle DMG B, Herald options, and so on (see the header of `tulu/rl/vec_env.py`).
- Reward: the multiplier gained at each step, divided by 100 for training. γ = 1 (finite horizon).

## Rule highlights (full version: `docs/model_spec.md`, in Chinese)

- Deck: 45 Asterisms (15 kinds × 3 copies). Each turn draws 4 Asterisms plus 2 random Runes. Asterisms are free; placing the 3rd one triggers Stars Surge automatically and ends the turn.
- **Runes may be played at any time before the 3rd Asterism**: before Recast you can first place your best two Asterisms in the Orbit to bank their effects; before Alter / Gleam / Dim you can lock good cards first.
- Aftershock: each unused Rune discarded at turn end gives +1 Etched Memory (cap 50). Other unused Runes discarded by Recast also trigger Aftershock (confirmed 2026-11-20).
- A card gained while the hand is full (12) cannot enter the hand.

## Known limitations

- The parametric policy only expresses what its scoring formula covers, so it gives a lower bound; rollout lookahead and the networks are used to check this.
- "Optimal" means optimal for the current model assumptions. The damage formula is expected to receive a major update; rerun `scripts/e_pipeline.py` afterwards.
- In parallel mode each thread has its own random-number state, so runs are not step-by-step reproducible. Use `VecEnv(..., parallel=False)` when reproducibility matters; single evaluations of the same network can differ by about ±0.3.
- No exact dynamic-programming check on a reduced instance has been done yet.

> **Live demo**: https://coliner-jiajie.github.io/tulu-nativity/ (GitHub Pages, serves `viz/`; set the Pages source to "GitHub Actions" in the repository settings)

## Replay visualization (`viz/`, static web page)

An interactive card animation of the trained policies (rollout expert / approach A / approach B PPO), where you can also **play it yourself** and compare against the program afterwards. It is a pure front-end page: the rules engine and the policies are ported to JavaScript (`viz/engine.js`), so **no server is needed**; double-click `viz/index.html` or host it on any static host. The page has a Chinese / English switch.

- **Replay mode**: choose a policy and a seed; step through drawing, Orbit resolution, Rune plays and Stars Surge. The right panel shows candidate scores for each decision.
- **Play mode**: choose a seed (same seed = same opening) and play cards yourself; "expert hint" only advises. At the end you get a same-seed comparison and the distribution of the program's repeated games on that opening, with your percentile.

| File | Purpose |
|---|---|
| `viz/index.html`, `viz/i18n.js` | page, animation, Chinese / English strings |
| `viz/engine.js` | rules engine + approach A + rollout expert + PPO inference (mirrors `tulu/env/core.py`, `tulu/param/`, `tulu/rl/expert.py`) |
| `viz/params.js` | environment config, approach-A θ, PPO weights, real-damage conversion (generated by `scripts/v_export_params.py`) |
| `viz/data.js` | pre-generated replays (generated by `scripts/v_gen_data.sh`, about a minute, using macOS's built-in jsc) |

After changing rules, config, policies or models, regenerate:

```bash
python scripts/v_export_params.py      # update viz/params.js
./scripts/v_gen_data.sh                # update viz/data.js
```
