// [可视化] 用 JS 引擎生成回放数据 viz/data.js（种子 1~N，三种策略）。macOS 自带的 jsc 即可运行，无需 Node：
//   cd viz && /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc ../scripts/v_gen_data.js > data.js
// 也可以用 scripts/v_gen_data.sh 一键生成。浏览器里换种子时会实时计算，这份数据只是让常用种子秒开。
var window = globalThis;
load("params.js"); load("engine.js");
const E = TuluEngine, P = TULU_PARAMS, C = P.cfg;
const N = 40;
const POL = {
  expert: { name: "滚动前瞻专家", short: "专家", mean: 1291.0, nameEn: "Rollout Expert",
    desc: "目前最强。每个决策点对每个候选动作各模拟 200 局（后续按方法 A 打），选平均总倍率最高的。约 1291。",
    descEn: "Strongest so far. At every decision it simulates 200 games per candidate move (the rest played with Method A) and picks the highest average total multiplier. About 1291." },
  param: { name: "方法 A · 参数化最优", short: "参数化", mean: 1229.9, nameEn: "Method A · Parametric",
    desc: "15 个参数的打分公式 + 一步前瞻，用 CEM 搜索最优参数；30 万局评估 1229.9。",
    descEn: "A scoring formula with 15 parameters plus one-step lookahead, optimized with CEM; 1229.9 over 300,000 games." },
  ppo: { name: "方法 B · PPO 神经网络", short: "PPO", mean: 1196.0, nameEn: "Method B · PPO neural network",
    desc: "纯强化学习的演员-评论家网络（贪心动作），2000 轮训练；约 1196。",
    descEn: "A pure reinforcement-learning actor-critic network (greedy actions), trained for 2000 rounds; about 1196." },
};
const episodes = [], pols = [];
for (const p of ["expert", "param", "ppo"]) {
  if (p === "ppo" && !E.getModel()) continue;
  const tot = [];
  for (let s = 1; s <= N; s++) {
    const ep = E.runSync(E.runPolicyG(p, s, { R: 200 }));
    episodes.push(ep); tot.push(ep.total);
  }
  pols.push(Object.assign({ key: p, n: tot.length, sample_mean: Math.round((tot.reduce((a, b) => a + b, 0) / tot.length + P.nuke) * 10) / 10 }, POL[p]));
  printErr(p + " 完成，样本均值 " + pols[pols.length - 1].sample_mean);
}
const meta = { nuke: P.nuke, rho: Math.round(C[10] * 1e4) / 1e4, h: Math.round(C[12] * 1e3) / 1e3, m: C[11], I0: C[1], lawcost: C[2], cap: C[3],
  fZ: C[4], fH: C[5], fY: C[6], policies: pols, real: P.real, engine: "js" };
print("window.TULU_DATA = " + JSON.stringify({ meta, episodes }) + ";");
