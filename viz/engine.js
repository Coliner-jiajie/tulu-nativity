/* 诞临·图鲁 诞临阶段：规则引擎 + 策略（纯 JavaScript 移植，浏览器与 jsc/Node 通用）。
 * 与 Python 版（tulu/env/core.py、tulu/param/policy.py、tulu/param/rollout.py、tulu/rl/expert.py、tulu/viz/recorder.py）
 * 一一对应，数组布局与函数名保持一致，便于对照。随机数用带种子的 mulberry32，
 * 所以「同一个种子 = 同一个开局」在任何策略、任何模式下都成立。
 * 依赖 params.js 提供 TULU_PARAMS（cfg/theta/真实伤害换算）与 TULU_MODEL（PPO 权重，可选）。 */
(function (root) {
"use strict";
const NT = 15, NL = 7;
const STAR_TYPE = ["灾", "垣", "蕴"];
const LAW_NAMES = ["易象", "增辉", "敛芒", "昭示", "引辰", "拓途", "重衍"];
const STAR_NAMES = []; for (let k = 0; k < 3; k++) for (let r = 1; r <= 5; r++) STAR_NAMES.push(STAR_TYPE[k] + r);
// ---- 双语文案：bi(fn) 把同一个模板分别以中文 / 英文各求值一次，得到 [中文, 英文] ----
const STAR_EN = ["Antares", "Scutum", "Pleiades"], LAW_EN = ["Alter", "Gleam", "Dim", "Herald", "Beckon", "Expand", "Recast"], ROMAN_EN = ["I", "II", "III", "IV", "V"];
let _L = 0;
const lg = (z, e) => (_L ? e : z);
const sn = c => _L ? STAR_EN[(c / 5) | 0] + " " + ROMAN_EN[c % 5] : STAR_NAMES[c];
const ln = l => _L ? LAW_EN[l] : LAW_NAMES[l];
const kn = k => _L ? STAR_EN[k] : STAR_TYPE[k];
function bi(fn) { _L = 0; const z = fn(); _L = 1; const e = fn(); _L = 0; return [z, e]; }
const err = (z, e) => new Error(root.TULU_LANG === "en" ? e : z);

// ---- 配置向量下标 ----
let ci = 0;
const C_N = ci++, C_I0 = ci++, C_LAWCOST = ci++, C_CAP = ci++, C_FZ = ci++, C_FH = ci++, C_FY = ci++, C_BASEHIT = ci++,
  C_PERZ = ci++, C_Y3 = ci++, C_RHO = ci++, C_M = ci++, C_H = ci++, C_MODE = ci++, C_FIRSTGROW = ci++, C_KAPPA = ci++,
  C_CARDB = ci++, C_TIDEB = ci++, C_B0 = ci++, C_GROW = ci++, C_NDRAW = ci++, C_NLAW = ci++, C_IMAX = ci++, C_AFTER = ci++;

// ---- 状态数组布局 ----
const O_HN = 0, O_HO = 1, O_HC = 17, O_LN = 33, O_LI = 34, O_LC = 50, O_TN = 66, O_TC = 67, O_TO = 70, O_DP = 73,
  O_DC = 88, O_I = 103, O_T = 104, O_PN = 105, O_PO = 106, O_COMMIT = 109, O_DONE = 110, O_PLAN = 111, SI_LEN = 120;
const F_B = 0, F_R = 1, SF_LEN = 4;
const ST_TOTAL = 0, ST_NLAW = 1, ST_LAWCNT = 2, ST_TURNR = 9, ST_N3Z = 15, ST_N3Y = 16, ST_N3H = 17, ST_BFIN = 18,
  ST_ILEFT = 19, ST_TURNY = 22, ST_TURNHITS = 28, ST_TURNZ = 34, ST_REFUND = 40, ST_CYBANK = 41, ST_LOCKLAW = 42, ST_LEN = 48;
const T_MU = 0, T_PZ = 1, T_B = 2, T_PHI = 9, THETA_LEN = 15;

// ---- 随机数：mulberry32（种子先散列一次，避免相邻种子相关） ----
let _s = 1;
function seedRng(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b); x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35); x ^= x >>> 16;
  _s = (x >>> 0) || 1;
}
function rnd() {
  _s = (_s + 0x6D2B79F5) >>> 0;
  let t = _s;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const randint = n => Math.floor(rnd() * n);

function fOf(k, cfg) { return k === 0 ? cfg[C_FZ] : k === 1 ? cfg[C_FH] : cfg[C_FY]; }

// ---- 星潮奔涌 ----
const TD = { dmg: 0, nZ: 0, nH: 0, nY: 0, hits: 0 };
function tideCalc(c0, c1, c2, B, cfg) {
  const k0 = (c0 / 5) | 0, k1 = (c1 / 5) | 0, k2 = (c2 / 5) | 0;
  const r0 = c0 % 5, r1 = c1 % 5, r2 = c2 % 5;
  let hits = 3;
  if (r0 === r1 && r1 === r2) hits += 6;
  else {
    const lo = Math.min(r0, r1, r2), hi = Math.max(r0, r1, r2), mid = r0 + r1 + r2 - lo - hi;
    if (mid - lo === 1 && hi - mid === 1) hits += 3;
  }
  if (k0 === k1 && k1 === k2) hits += 6;
  else if (k0 !== k1 && k1 !== k2 && k0 !== k2) hits += 3;
  const nZ = (k0 === 0) + (k1 === 0) + (k2 === 0), nH = (k0 === 1) + (k1 === 1) + (k2 === 1), nY = (k0 === 2) + (k1 === 2) + (k2 === 2);
  let per = cfg[C_BASEHIT] + cfg[C_PERZ] * nZ;
  if (nZ === 3) { if (cfg[C_MODE] < 0.5) per += cfg[C_KAPPA] * B; else per *= 1.0 + cfg[C_KAPPA] * B; }
  let dmg = hits * per;
  if (nY === 3) dmg *= cfg[C_Y3];
  TD.dmg = dmg; TD.nZ = nZ; TD.nH = nH; TD.nY = nY; TD.hits = hits;
  return dmg;
}

// ---- 抽牌与手牌操作 ----
function drawOne(si) {
  let tot = 0;
  for (let c = 0; c < NT; c++) tot += si[O_DP + c];
  if (tot === 0) {
    for (let c = 0; c < NT; c++) { si[O_DP + c] = si[O_DC + c]; si[O_DC + c] = 0; tot += si[O_DP + c]; }
    if (tot === 0) return -1;
  }
  let r = randint(tot);
  for (let c = 0; c < NT; c++) { r -= si[O_DP + c]; if (r < 0) { si[O_DP + c] -= 1; return c; } }
  return -1;
}
function addStar(si, c, cap) {
  if (c < 0) return;
  const n = si[O_HN];
  if (n + si[O_LN] < cap) { si[O_HO + n] = c; si[O_HC + n] = c; si[O_HN] = n + 1; }
  else si[O_DC + c] += 1;
}
function addLaw(si, lid, cost, cap) {
  const n = si[O_LN];
  if (si[O_HN] + n < cap) { si[O_LI + n] = lid; si[O_LC + n] = cost; si[O_LN] = n + 1; }
}
const _arr = new Int32Array(NL);
function pickDistinct(nTotal, nPick, out) {
  for (let i = 0; i < nTotal; i++) _arr[i] = i;
  for (let i = 0; i < nPick; i++) {
    const j = i + randint(nTotal - i);
    const tmp = _arr[i]; _arr[i] = _arr[j]; _arr[j] = tmp;
    out[i] = _arr[i];
  }
}
const _out8 = new Int32Array(8);
function genLaws(si, cfg, n) {
  pickDistinct(NL, n, _out8);
  for (let i = 0; i < n; i++) addLaw(si, _out8[i], cfg[C_LAWCOST] | 0, cfg[C_CAP] | 0);
}
function startTurn(si, sf, cfg) {
  const t = si[O_T];
  if (t > 1 || cfg[C_FIRSTGROW] > 0.5) sf[F_B] += cfg[C_GROW] * cfg[C_H] * cfg[C_M];
  si[O_HN] = 0; si[O_LN] = 0; si[O_TN] = 0; si[O_COMMIT] = 0; si[O_PN] = 0;
  const cap = cfg[C_CAP] | 0;
  for (let i = 0; i < (cfg[C_NDRAW] | 0); i++) addStar(si, drawOne(si), cap);
  genLaws(si, cfg, cfg[C_NLAW] | 0);
}
function endTurn(si, sf, cfg, st) {
  const gain = (cfg[C_AFTER] | 0) * si[O_LN], imax = cfg[C_IMAX] | 0;
  if (gain > 0 && si[O_I] < imax) {
    const ni = Math.min(imax, si[O_I] + gain);
    st[ST_REFUND] += ni - si[O_I]; si[O_I] = ni;
  }
  for (let i = 0; i < si[O_HN]; i++) si[O_DC + si[O_HO + i]] += 1;
  for (let i = 0; i < si[O_TN]; i++) si[O_DC + si[O_TO + i]] += 1;
  si[O_HN] = 0; si[O_LN] = 0; si[O_TN] = 0;
  si[O_T] += 1;
  if (si[O_T] > (cfg[C_N] | 0)) si[O_DONE] = 1;
  else startTurn(si, sf, cfg);
}
function initGame(si, sf, cfg) {
  si.fill(0); sf.fill(0);
  for (let c = 0; c < NT; c++) si[O_DP + c] = 3;
  si[O_I] = cfg[C_I0] | 0; si[O_T] = 1;
  sf[F_B] = cfg[C_B0] * cfg[C_H];
  startTurn(si, sf, cfg);
}
function placeStar(si, sf, cfg, hi, st) {
  const cur = si[O_HC + hi], orig = si[O_HO + hi], n = si[O_HN];
  si[O_HC + hi] = si[O_HC + n - 1]; si[O_HO + hi] = si[O_HO + n - 1]; si[O_HN] = n - 1;
  const j = si[O_TN] + 1, k = (cur / 5) | 0;
  sf[F_R] += fOf(k, cfg) * j;
  st[ST_TURNR + si[O_T] - 1] += fOf(k, cfg) * j;
  if (k === 2) sf[F_B] += cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * j;
  si[O_TC + si[O_TN]] = cur; si[O_TO + si[O_TN]] = orig; si[O_TN] = j;
  if (j === 3) {
    const dmg = tideCalc(si[O_TC], si[O_TC + 1], si[O_TC + 2], sf[F_B], cfg);
    const nZ = TD.nZ, nH = TD.nH, nY = TD.nY, hits = TD.hits;
    sf[F_R] += dmg;
    const t = si[O_T] - 1;
    st[ST_TURNR + t] += dmg; st[ST_TURNY + t] = nY; st[ST_TURNZ + t] = nZ; st[ST_TURNHITS + t] = hits;
    if (nZ === 3) st[ST_N3Z] += 1;
    if (nH === 3) st[ST_N3H] += 1;
    if (nY === 3) st[ST_N3Y] += 1;
    sf[F_B] += cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY;
    endTurn(si, sf, cfg, st);
  }
}
function playLaw(si, sf, cfg, li, st) {
  const lid = si[O_LI + li], cost = si[O_LC + li];
  si[O_I] -= cost;
  const n = si[O_LN];
  si[O_LI + li] = si[O_LI + n - 1]; si[O_LC + li] = si[O_LC + n - 1]; si[O_LN] = n - 1;
  st[ST_NLAW] += 1; st[ST_LAWCNT + lid] += 1;
  if (si[O_TN] > 0) { st[ST_LOCKLAW] += 1; if (lid === 6) st[ST_CYBANK] += 1; }
  const cap = cfg[C_CAP] | 0;
  if (lid === 0) {            // 易象
    for (let i = 0; i < si[O_HN]; i++) {
      const c = si[O_HC + i];
      const nk = (((c / 5) | 0) + 1 + randint(2)) % 3;
      si[O_HC + i] = nk * 5 + c % 5;
    }
  } else if (lid === 1) {     // 增辉
    for (let i = 0; i < si[O_HN]; i++) { const c = si[O_HC + i]; si[O_HC + i] = ((c / 5) | 0) * 5 + Math.min(4, c % 5 + 1); }
  } else if (lid === 2) {     // 敛芒
    for (let i = 0; i < si[O_HN]; i++) { const c = si[O_HC + i]; si[O_HC + i] = ((c / 5) | 0) * 5 + Math.max(0, c % 5 - 1); }
  } else if (lid === 3) {     // 昭示
    pickDistinct(NL, 3, _out8);
    for (let i = 0; i < 3; i++) si[O_PO + i] = _out8[i];
    si[O_PN] = 3;
  } else if (lid === 4) {     // 引辰
    addStar(si, drawOne(si), cap); addStar(si, drawOne(si), cap);
  } else if (lid === 5) {     // 拓途
    pickDistinct(NL, 3, _out8);
    for (let i = 0; i < 3; i++) addLaw(si, _out8[i], cfg[C_LAWCOST] | 0, cap);
  } else {                    // 重衍：丢弃的未使用法则触发余波
    {
      const gain = (cfg[C_AFTER] | 0) * si[O_LN], imax = cfg[C_IMAX] | 0;
      if (gain > 0 && si[O_I] < imax) {
        const ni = Math.min(imax, si[O_I] + gain);
        st[ST_REFUND] += ni - si[O_I]; si[O_I] = ni;
      }
    }
    for (let i = 0; i < si[O_TN]; i++) si[O_DC + si[O_TO + i]] += 1;
    si[O_TN] = 0;
    for (let i = 0; i < si[O_HN]; i++) si[O_DC + si[O_HO + i]] += 1;
    si[O_HN] = 0; si[O_LN] = 0;
    for (let i = 0; i < 4; i++) addStar(si, drawOne(si), cap);
    genLaws(si, cfg, 2);
  }
}
function pickOption(si, cfg, j) {
  addLaw(si, si[O_PO + j], 0, cfg[C_CAP] | 0);
  si[O_PN] = 0;
}

// =========================================================================
// 方法 A：参数化策略（移植 tulu/param/policy.py）
// =========================================================================
function scoreFinal(c0, c1, c2, tn, B, t, cfg, theta) {
  const cb = cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M];
  let card = 0.0, dB = 0.0;
  if (tn <= 0) { card += fOf((c0 / 5) | 0, cfg) * 1; if (((c0 / 5) | 0) === 2) dB += cb * 1; }
  if (tn <= 1) { card += fOf((c1 / 5) | 0, cfg) * 2; if (((c1 / 5) | 0) === 2) dB += cb * 2; }
  card += fOf((c2 / 5) | 0, cfg) * 3;
  if (((c2 / 5) | 0) === 2) dB += cb * 3;
  const dmg = tideCalc(c0, c1, c2, B, cfg);
  dB += cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * TD.nY;
  let score = card + dmg + theta[T_PHI + t - 1] * dB;
  if (TD.nZ < 3) score -= theta[T_PZ] * TD.nZ;
  return score;
}
function bestCompletion(hc, hn, trc, tn, B, t, cfg, theta, plan) {
  const s = 3 - tn;
  let best = -1e18;
  if (hn < s) return best;
  if (s === 3) {
    for (let i = 0; i < hn; i++) for (let j = i + 1; j < hn; j++) for (let k = j + 1; k < hn; k++) {
      let a = hc[i], b = hc[j], c = hc[k];
      let fa = fOf((a / 5) | 0, cfg), fb = fOf((b / 5) | 0, cfg), fc = fOf((c / 5) | 0, cfg), x;
      if (fa > fb) { x = a; a = b; b = x; x = fa; fa = fb; fb = x; }
      if (fb > fc) { x = b; b = c; c = x; x = fb; fb = fc; fc = x; }
      if (fa > fb) { x = a; a = b; b = x; x = fa; fa = fb; fb = x; }
      const sc = scoreFinal(a, b, c, 0, B, t, cfg, theta);
      if (sc > best) { best = sc; plan[0] = a; plan[1] = b; plan[2] = c; }
    }
  } else if (s === 2) {
    const p0 = trc[0];
    for (let i = 0; i < hn; i++) for (let j = i + 1; j < hn; j++) {
      let a = hc[i], b = hc[j];
      if (fOf((a / 5) | 0, cfg) > fOf((b / 5) | 0, cfg)) { const x = a; a = b; b = x; }
      const sc = scoreFinal(p0, a, b, 1, B, t, cfg, theta);
      if (sc > best) { best = sc; plan[1] = a; plan[2] = b; }
    }
  } else {
    const p0 = trc[0], p1 = trc[1];
    for (let i = 0; i < hn; i++) {
      const sc = scoreFinal(p0, p1, hc[i], 2, B, t, cfg, theta);
      if (sc > best) { best = sc; plan[2] = hc[i]; }
    }
  }
  return best;
}
function sampleDraw(dpc) {
  let tot = 0;
  for (let c = 0; c < NT; c++) tot += dpc[c];
  if (tot === 0) return -1;
  let r = randint(tot);
  for (let c = 0; c < NT; c++) { r -= dpc[c]; if (r < 0) { dpc[c] -= 1; return c; } }
  return -1;
}
function lockValue(card, slot, cfg, theta, t) {
  let v = fOf((card / 5) | 0, cfg) * slot;
  if (((card / 5) | 0) === 2) v += theta[T_PHI + t - 1] * cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * slot;
  return v;
}
const _plan = new Int32Array(3), _tmp = new Int32Array(16), _trc2 = new Int32Array(3), _dpc = new Int32Array(NT), _hand2 = new Int32Array(16);
function lawValue2(lid, cost, si, B, t, cfg, theta, K, V0, out) {
  out[0] = -1; out[1] = -1;
  const hn = si[O_HN], tn = si[O_TN];
  const hc = si.subarray(O_HC, O_HC + 16), trc = si.subarray(O_TC, O_TC + 3);
  const base = theta[T_B + lid] - theta[T_MU] * cost;
  const plan = _plan, tmp = _tmp, trc2 = _trc2, dpc = _dpc;
  const cap = cfg[C_CAP] | 0;
  const room0 = cap - (hn + si[O_LN] - 1);
  if (lid === 3 || lid === 5) return V0 + base;
  if (lid === 4) {
    let tot = 0.0;
    for (let s = 0; s < K; s++) {
      for (let i = 0; i < hn; i++) tmp[i] = hc[i];
      for (let c = 0; c < NT; c++) dpc[c] = si[O_DP + c];
      let n2 = hn, room = room0;
      for (let q = 0; q < 2; q++) { const d = sampleDraw(dpc); if (d >= 0 && room > 0) { tmp[n2] = d; n2++; room--; } }
      let v = bestCompletion(tmp, n2, trc, tn, B, t, cfg, theta, plan);
      if (v < -1e17) v = -1e6;
      tot += v;
    }
    return tot / K + base;
  }
  if (lid === 6) {
    const nb = 2 - tn;
    let lockval = 0.0, i1 = -1, i2 = -1;
    for (let i = 0; i < hn; i++) {
      const f = fOf((hc[i] / 5) | 0, cfg);
      if (i1 < 0 || f > fOf((hc[i1] / 5) | 0, cfg)) { i2 = i1; i1 = i; }
      else if (i2 < 0 || f > fOf((hc[i2] / 5) | 0, cfg)) i2 = i;
    }
    if (nb === 2 && i1 >= 0 && i2 >= 0) {
      lockval = lockValue(hc[i2], 1, cfg, theta, t) + lockValue(hc[i1], 2, cfg, theta, t);
      out[0] = hc[i2]; out[1] = hc[i1];
    } else if (nb === 2 && i1 >= 0) { lockval = lockValue(hc[i1], 1, cfg, theta, t); out[0] = hc[i1]; }
    else if (nb === 1 && i1 >= 0) { lockval = lockValue(hc[i1], 2, cfg, theta, t); out[0] = hc[i1]; }
    let tot = 0.0;
    for (let s = 0; s < K; s++) {
      for (let c = 0; c < NT; c++) dpc[c] = si[O_DP + c];
      let n2 = 0;
      for (let q = 0; q < 4; q++) { const d = sampleDraw(dpc); if (d >= 0) { tmp[n2] = d; n2++; } }
      let v = bestCompletion(tmp, n2, trc2.fill(0), 0, B, t, cfg, theta, plan);
      if (v < -1e17) v = -1e6;
      tot += v;
    }
    return lockval + tot / K + base;
  }
  // 易象 / 增辉 / 敛芒：枚举「先锁定若干张」的方案
  let bestVal = -1e18;
  const hand2 = _hand2;
  const nlockMax = 2 - tn;
  let nopt = 1;
  if (nlockMax >= 1) nopt += hn;
  if (nlockMax >= 2) nopt += hn * (hn - 1);
  for (let opt = 0; opt < nopt; opt++) {
    let a = -1, b = -1;
    if (opt >= 1 && opt <= hn && nlockMax >= 1) a = opt - 1;
    else if (opt > hn) {
      const q = opt - 1 - hn;
      a = Math.floor(q / (hn - 1));
      const r = q % (hn - 1);
      b = r < a ? r : r + 1;
    }
    let lockval = 0.0, n2 = 0;
    for (let i = 0; i < 3; i++) trc2[i] = trc[i];
    let tn2 = tn;
    if (a >= 0) { lockval += lockValue(hc[a], tn2 + 1, cfg, theta, t); trc2[tn2] = hc[a]; tn2++; }
    if (b >= 0) { lockval += lockValue(hc[b], tn2 + 1, cfg, theta, t); trc2[tn2] = hc[b]; tn2++; }
    for (let i = 0; i < hn; i++) if (i !== a && i !== b) { hand2[n2] = hc[i]; n2++; }
    let val;
    if (lid === 1 || lid === 2) {
      for (let i = 0; i < n2; i++) {
        const c = hand2[i];
        tmp[i] = lid === 1 ? ((c / 5) | 0) * 5 + Math.min(4, c % 5 + 1) : ((c / 5) | 0) * 5 + Math.max(0, c % 5 - 1);
      }
      const v = bestCompletion(tmp, n2, trc2, tn2, B, t, cfg, theta, plan);
      if (v < -1e17) continue;
      val = lockval + v + base;
    } else {
      let tot = 0.0, ok = true;
      for (let s = 0; s < K; s++) {
        for (let i = 0; i < n2; i++) {
          const c = hand2[i];
          const nk = (((c / 5) | 0) + 1 + randint(2)) % 3;
          tmp[i] = nk * 5 + c % 5;
        }
        const v = bestCompletion(tmp, n2, trc2, tn2, B, t, cfg, theta, plan);
        if (v < -1e17) { ok = false; break; }
        tot += v;
      }
      if (!ok) continue;
      val = lockval + tot / K + base;
    }
    if (val > bestVal) { bestVal = val; out[0] = a >= 0 ? hc[a] : -1; out[1] = b >= 0 ? hc[b] : -1; }
  }
  return bestVal;
}
function placeByValue(si, sf, cfg, cur, st) {
  for (let i = 0; i < si[O_HN]; i++) if (si[O_HC + i] === cur) { placeStar(si, sf, cfg, i, st); return true; }
  return false;
}
const _p0 = new Int32Array(3), _o = new Int32Array(2), _ob = new Int32Array(2);
function playUntilDone(si, sf, theta, cfg, K, mode, st) {
  let guard = 0;
  const plan0 = _p0, out = _o, outb = _ob;
  while (si[O_DONE] === 0 && guard < 2000) {
    guard++;
    const t = si[O_T];
    if (mode === 1) {         // 均匀随机合法动作
      if (si[O_PN] > 0) { pickOption(si, cfg, randint(si[O_PN])); continue; }
      let nl = 0;
      for (let li = 0; li < si[O_LN]; li++) if (si[O_I] >= si[O_LC + li]) nl++;
      const ns = si[O_HN];
      if (nl + ns === 0) break;
      const r = randint(nl + ns);
      if (r < nl) {
        let cnt = 0;
        for (let li = 0; li < si[O_LN]; li++) if (si[O_I] >= si[O_LC + li]) { if (cnt === r) { playLaw(si, sf, cfg, li, st); break; } cnt++; }
      } else placeStar(si, sf, cfg, r - nl, st);
      continue;
    }
    const hn = si[O_HN], tn = si[O_TN];
    const V0 = bestCompletion(si.subarray(O_HC, O_HC + 16), hn, si.subarray(O_TC, O_TC + 3), tn, sf[F_B], t, cfg, theta, plan0);
    if (si[O_PN] > 0) {
      let bv = -1e18, bj = 0;
      for (let j = 0; j < si[O_PN]; j++) {
        const v = lawValue2(si[O_PO + j], 0, si, sf[F_B], t, cfg, theta, K, V0, out);
        if (v > bv) { bv = v; bj = j; }
      }
      pickOption(si, cfg, bj);
      continue;
    }
    if (si[O_COMMIT] === 1) { placeByValue(si, sf, cfg, si[O_PLAN + si[O_TN]], st); continue; }
    let best = V0, bestl = -1;
    outb[0] = -1; outb[1] = -1;
    for (let li = 0; li < si[O_LN]; li++) {
      const lid = si[O_LI + li], cost = si[O_LC + li];
      if (si[O_I] < cost) continue;
      let dup = false;
      for (let lj = 0; lj < li; lj++) if (si[O_LI + lj] === lid && si[O_LC + lj] === cost) dup = true;
      if (dup) continue;
      const v = lawValue2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out);
      if (v > best + 1e-9) { best = v; bestl = li; outb[0] = out[0]; outb[1] = out[1]; }
    }
    if (bestl >= 0) {
      for (let q = 0; q < 2; q++) if (outb[q] >= 0) placeByValue(si, sf, cfg, outb[q], st);
      playLaw(si, sf, cfg, bestl, st);
    } else {
      if (V0 < -1e17) break;
      si[O_PLAN] = plan0[0]; si[O_PLAN + 1] = plan0[1]; si[O_PLAN + 2] = plan0[2];
      si[O_COMMIT] = 1;
    }
  }
  st[ST_TOTAL] = sf[F_R]; st[ST_BFIN] = sf[F_B]; st[ST_ILEFT] = si[O_I];
}

// =========================================================================
// 滚动前瞻（移植 tulu/param/rollout.py 与 tulu/rl/expert.py）
// =========================================================================
const KIND_FINISH = 0, KIND_LAW = 1, KIND_OPTION = 2;
function applyAction(si, sf, cfg, kind, a, l1, l2, p1, p2, p3, st) {
  if (kind === KIND_LAW) {
    if (l1 >= 0) placeByValue(si, sf, cfg, l1, st);
    if (l2 >= 0) placeByValue(si, sf, cfg, l2, st);
    playLaw(si, sf, cfg, a, st);
  } else if (kind === KIND_OPTION) pickOption(si, cfg, a);
  else { si[O_PLAN] = p1; si[O_PLAN + 1] = p2; si[O_PLAN + 2] = p3; si[O_COMMIT] = 1; }
}
const _si2 = new Int32Array(SI_LEN), _sf2 = new Float64Array(SF_LEN), _st2 = new Float64Array(48);
function actionValue(si, sf, theta, cfg, K, R, seedBase, kind, a, l1, l2, p1, p2, p3) {
  let tot = 0.0;
  for (let r = 0; r < R; r++) {
    _si2.set(si); _sf2.set(sf); _st2.fill(0);
    seedRng(seedBase + r);
    applyAction(_si2, _sf2, cfg, kind, a, l1, l2, p1, p2, p3, _st2);
    playUntilDone(_si2, _sf2, theta, cfg, K, 0, _st2);
    tot += _sf2[F_R];
  }
  return tot / R;
}

// =========================================================================
// PPO 网络推理（移植 tulu/rl/networks.py 与 vec_env.build_obs_mask）
// =========================================================================
const NUM_ACTIONS = 36, OBS_DIM = 94, A_LAW0 = 15, A_FREE0 = 22, A_OPT0 = 29;
function buildObsMask(si, sf, cfg, obs, mask) {
  obs.fill(0); mask.fill(false);
  for (let i = 0; i < si[O_HN]; i++) obs[si[O_HC + i]] += 0.25;
  for (let i = 0; i < si[O_TN]; i++) obs[15 + si[O_TC + i]] += 1 / 3;
  obs[30 + Math.min(si[O_TN], 2)] = 1.0;
  for (let li = 0; li < si[O_LN]; li++) { const lid = si[O_LI + li]; if (si[O_LC + li] > 0) obs[33 + lid] += 0.5; else obs[40 + lid] += 0.5; }
  for (let c = 0; c < NT; c++) { obs[47 + c] = si[O_DP + c] / 3; obs[62 + c] = si[O_DC + c] / 3; }
  obs[77] = si[O_I] / 50;
  const t = si[O_T];
  if (t >= 1 && t <= 6) obs[78 + t - 1] = 1.0;
  obs[84] = sf[F_B] * 2;
  for (let j = 0; j < si[O_PN]; j++) obs[85 + si[O_PO + j]] = 1.0;
  obs[92] = (si[O_HN] + si[O_LN]) / 12; obs[93] = si[O_LN] / 12;
  if (si[O_PN] > 0) { for (let j = 0; j < si[O_PN]; j++) mask[A_OPT0 + si[O_PO + j]] = true; }
  else {
    for (let i = 0; i < si[O_HN]; i++) mask[si[O_HC + i]] = true;
    for (let li = 0; li < si[O_LN]; li++) {
      const lid = si[O_LI + li], cost = si[O_LC + li];
      if (si[O_I] >= cost) { if (cost > 0) mask[A_LAW0 + lid] = true; else mask[A_FREE0 + lid] = true; }
    }
  }
}
function linear(x, W, b, nIn, nOut, relu) {
  const y = new Float64Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = b[o]; const base = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[base + i] * x[i];
    y[o] = relu && s < 0 ? 0 : s;
  }
  return y;
}
function ppoProbs(model, obs, mask) {
  const h = model.hidden;
  let x = linear(obs, model.W1, model.b1, OBS_DIM, h, true);
  x = linear(x, model.W2, model.b2, h, h, true);
  const lg = linear(x, model.Wp, model.bp, h, NUM_ACTIONS, false);
  let mx = -1e30;
  for (let i = 0; i < NUM_ACTIONS; i++) { if (!mask[i]) lg[i] = -1e9; if (lg[i] > mx) mx = lg[i]; }
  let sum = 0;
  for (let i = 0; i < NUM_ACTIONS; i++) { lg[i] = Math.exp(lg[i] - mx); sum += lg[i]; }
  for (let i = 0; i < NUM_ACTIONS; i++) lg[i] /= sum;
  return lg;
}
const ACTION_NAMES = [];
for (let c = 0; c < 15; c++) ACTION_NAMES.push(bi(() => lg("放入 ", "Place ") + sn(c)));
for (let l = 0; l < 7; l++) ACTION_NAMES.push(bi(() => lg("打出 ", "Play ") + ln(l)));
for (let l = 0; l < 7; l++) ACTION_NAMES.push(bi(() => lg("打出 " + ln(l) + "（0 消耗）", "Play " + ln(l) + " (cost 0)")));
for (let l = 0; l < 7; l++) ACTION_NAMES.push(bi(() => lg("昭示选取 ", "Herald pick ") + ln(l)));

// =========================================================================
// 带帧录制的对局（移植 tulu/viz/recorder.py 的 Game）
// =========================================================================
const r3 = x => Math.round(x * 1000) / 1000;
const f1 = x => x.toFixed(1);
const gfmt = x => String(+x.toPrecision(12));

class Game {
  constructor(cfg, seed) {
    this.cfg = cfg; this.seed = Math.floor(seed);
    this.si = new Int32Array(SI_LEN); this.sf = new Float64Array(SF_LEN); this.st = new Float64Array(ST_LEN);
    seedRng(this.seed);
    initGame(this.si, this.sf, cfg);
    this._n = 0; this.nact = 0;
    this.hand = []; this.laws = []; this.track = [];
    this.frames = []; this.decs = []; this.cur_d = null; this.d_new = false;
    this.newTurnIds();
    this.emit("start", this.startText());
  }
  nid() { return ++this._n; }
  done() { return !!this.si[O_DONE]; }
  handList() { const si = this.si, o = []; for (let i = 0; i < si[O_HN]; i++) o.push([this.hand[i], si[O_HC + i], si[O_HO + i]]); return o; }
  lawsList() { const si = this.si, o = []; for (let i = 0; i < si[O_LN]; i++) o.push([this.laws[i], si[O_LI + i], si[O_LC + i]]); return o; }
  trackList() { const si = this.si, o = []; for (let i = 0; i < si[O_TN]; i++) o.push([this.track[i], si[O_TC + i], si[O_TO + i]]); return o; }
  snap(o = {}) {
    const si = this.si, sf = this.sf;
    const opts = []; for (let j = 0; j < si[O_PN]; j++) opts.push(si[O_PO + j]);
    return {
      t: o.t !== undefined ? o.t : Math.min(si[O_T], this.cfg[C_N]),
      h: o.hand || this.handList(), l: o.laws || this.lawsList(), k: o.track || this.trackList(),
      o: o.opts || opts, I: o.I !== undefined ? o.I : si[O_I],
      B: Math.round((o.B !== undefined ? o.B : sf[F_B]) * 1e4) / 1e4, R: r3(o.R !== undefined ? o.R : sf[F_R]),
      dp: o.dp || Array.from(si.subarray(O_DP, O_DP + NT)), dc: o.dc || Array.from(si.subarray(O_DC, O_DC + NT)),
    };
  }
  emit(ev, txt, snap, extra) {
    const f = snap || this.snap();
    f.ev = ev; f.tx = txt[0]; f.te = txt[1];
    if (this.cur_d !== null) { f.d = this.cur_d; if (this.d_new) { f.df = 1; this.d_new = false; } }
    if (extra) Object.assign(f, extra);
    this.frames.push(f);
    return f;
  }
  decision(title, unit, cands, chosen) {
    this.decs.push({ t: this.si[O_T], ti: title[0], te: title[1], u: unit, c: cands.map(([lb, v], i) => [lb[0], r3(v), i === chosen ? 1 : 0, lb[1]]) });
    this.cur_d = this.decs.length - 1; this.d_new = true;
  }
  newTurnIds() {
    const si = this.si;
    this.hand = []; this.laws = []; this.track = [];
    for (let i = 0; i < si[O_HN]; i++) this.hand.push(this.nid());
    for (let i = 0; i < si[O_LN]; i++) this.laws.push(this.nid());
  }
  startText() {
    const t = this.si[O_T], B = this.sf[F_B];
    return bi(() => {
      let s = lg(`第 ${t} 回合开始：抽 4 张星象，随机获得 2 张法则`, `Turn ${t} begins: draw 4 Asterisms, gain 2 random Runes`);
      if (t > 1) s += lg(`；触腕伤害自然增长到 ${B.toFixed(3)}`, `; Tentacle DMG grows naturally to ${B.toFixed(3)}`);
      return s;
    });
  }
  placeForm(cur) {
    const si = this.si;
    for (let i = 0; i < si[O_HN]; i++) if (si[O_HC + i] === cur) return this.placeIdx(i);
    throw err("手中没有形态为 " + STAR_NAMES[cur] + " 的星象", "No such Asterism in hand: " + bi(() => sn(cur))[1]);
  }
  placeIdx(hi) {
    const { si, sf, cfg, st } = this;
    const cur = si[O_HC + hi], orig = si[O_HO + hi], cid = this.hand[hi];
    const j = si[O_TN] + 1, k = (cur / 5) | 0;
    const eff = fOf(k, cfg), dmg = eff * j;
    const dB = k === 2 ? cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * j : 0.0;
    const hand0 = this.handList(), laws0 = this.lawsList(), track0 = this.trackList();
    const handAfter = hand0.filter(h => h[0] !== cid);
    const trackAfter = track0.concat([[cid, cur, orig]]);
    const R0 = sf[F_R], B0 = sf[F_B], I0 = si[O_I], t = si[O_T];
    const dp0 = Array.from(si.subarray(O_DP, O_DP + NT)), dc0 = Array.from(si.subarray(O_DC, O_DC + NT));
    this.hand[hi] = this.hand[this.hand.length - 1]; this.hand.pop();
    this.track.push(cid);
    const txt = bi(() => {
      let x = lg(`${sn(cur)} 放入星轨第 ${j} 位，卡效触发 ${j} 次：${gfmt(eff)} × ${j} = ${f1(dmg)}`, `${sn(cur)} enters Orbit ${j} and triggers ${j} time${j > 1 ? "s" : ""}: ${gfmt(eff)} × ${j} = ${f1(dmg)}`);
      if (dB > 0) x += lg(`；触腕伤害 +${dB.toFixed(4)}`, `; Tentacle DMG +${dB.toFixed(4)}`);
      return x;
    });
    const ex = { cid, slot: j, dmg: r3(dmg), dB: Math.round(dB * 1e4) / 1e4, kind: k };
    if (j < 3) { placeStar(si, sf, cfg, hi, st); return this.emit("place", txt, null, ex); }
    // 第 3 张：卡效 → 星潮奔涌 → 回合结束
    const cs = [trackAfter[0][1], trackAfter[1][1], trackAfter[2][1]];
    const B1 = B0 + dB;
    const tdmg = tideCalc(cs[0], cs[1], cs[2], B1, cfg);
    const info = this.tideInfo(cs, B1, tdmg, TD.nZ, TD.nH, TD.nY, TD.hits);
    const nY = TD.nY;
    placeStar(si, sf, cfg, hi, st);
    const R1 = R0 + dmg;
    this.emit("place", txt, this.snap({ hand: handAfter, laws: laws0, track: trackAfter, I: I0, B: B1, R: R1, dp: dp0, dc: dc0, t, opts: [] }), ex);
    const turnR = st[ST_TURNR + t - 1];
    this.emit("tide", [info.txt, info.te], this.snap({ hand: handAfter, laws: laws0, track: trackAfter, I: I0, B: B1, R: R1 + tdmg, dp: dp0, dc: dc0, t, opts: [] }),
      { tide: info, tr: r3(turnR) });
    const dcE = dc0.slice();
    handAfter.forEach(h => { dcE[h[2]]++; }); trackAfter.forEach(q => { dcE[q[2]]++; });
    const gain = Math.min(cfg[C_IMAX] | 0, I0 + (cfg[C_AFTER] | 0) * laws0.length) - I0;
    const B2 = B1 + cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY;
    const etxt = bi(() => {
      let x = lg(`第 ${t} 回合结束：本回合共 ${f1(turnR)}；弃置手牌与星轨`, `Turn ${t} ends: ${f1(turnR)} this turn; hand and Orbits are discarded`);
      if (laws0.length) x += lg(`；${laws0.length} 张未用法则消散，余波使铭文 +${gain}`, `; ${laws0.length} unused Rune${laws0.length > 1 ? "s" : ""} fade, Aftershock: Etched Memory +${gain}`);
      return x;
    });
    this.emit("end", etxt, this.snap({ hand: [], laws: [], track: [], I: I0 + gain, B: B2, R: sf[F_R], dp: dp0, dc: dcE, t, opts: [] }), { gain, nlaw: laws0.length });
    if (this.done()) {
      this.cur_d = null;
      const Rt = sf[F_R]; this.emit("over", bi(() => lg(`诞临结束：6 回合合计 ${f1(Rt)}（另有「忘却前夜」+80）`, `Theophany ends: ${f1(Rt)} over 6 turns (plus Morimens +80)`)));
    } else {
      this.newTurnIds(); this.cur_d = null;
      this.emit("start", this.startText());
    }
  }
  tideInfo(cs, B, dmg, nZ, nH, nY, hits) {
    const cfg = this.cfg;
    const ks = cs.map(c => (c / 5) | 0), rs = cs.map(c => c % 5 + 1);
    let a = 0, an = ["", ""], b = 0, bn = ["", ""];
    const mn = Math.min(...rs);
    if (rs[0] === rs[1] && rs[1] === rs[2]) { a = 6; an = ["同星阶 +6", "Same Brilliance +6"]; }
    else if ([...rs].sort((x, y) => x - y).join() === [mn, mn + 1, mn + 2].join()) { a = 3; an = ["三连号 +3", "3 consecutive +3"]; }
    if (ks[0] === ks[1] && ks[1] === ks[2]) { b = 6; bn = ["同种类 +6", "Same type +6"]; }
    else if (new Set(ks).size === 3) { b = 3; bn = ["三种各一 +3", "3 different types +3"]; }
    const per = cfg[C_BASEHIT] + cfg[C_PERZ] * nZ;
    const bonus = nZ === 3 ? cfg[C_KAPPA] * B : 0.0;
    const txt = bi(() => {
      let x = lg(`星潮奔涌：${hits} 次 × ${(per + bonus).toFixed(2)} = ${f1(dmg)}`, `Stars Surge: ${hits} hits × ${(per + bonus).toFixed(2)} = ${f1(dmg)}`);
      if (nY === 3) x += lg(`（蕴蕴蕴 ×${gfmt(cfg[C_Y3])}）`, ` (3 Pleiades ×${gfmt(cfg[C_Y3])})`);
      return x;
    });
    return { hits, a, an: an[0], ane: an[1], b, bn: bn[0], bne: bn[1], per: r3(per), bonus: r3(bonus), dmg: r3(dmg), nZ, nH, nY, pat: ks.map(k => "灾垣蕴"[k]).join(""), cs, txt: txt[0], te: txt[1],
      y3: nY === 3 ? cfg[C_Y3] : 1.0, B: Math.round(B * 1e4) / 1e4 };
  }
  lawIdx(li) {
    const { si, sf, cfg, st } = this;
    const lid = si[O_LI + li], cost = si[O_LC + li], lawid = this.laws[li];
    const hand0 = this.handList(), hn0 = si[O_HN], ln0 = si[O_LN], locked = si[O_TN] > 0;
    this.laws[li] = this.laws[this.laws.length - 1]; this.laws.pop();
    playLaw(si, sf, cfg, li, st);
    const chg = [];
    let txt;
    const head = () => lg(`打出「${ln(lid)}」` + (cost ? `（消耗 ${cost} 层铭文）` : "（0 消耗）"), `Play ${ln(lid)}` + (cost ? ` (costs ${cost} Etched Memory)` : " (cost 0)"));
    if (lid <= 2) {
      hand0.forEach(([hid, c0], i) => { const c1 = si[O_HC + i]; if (c1 !== c0) chg.push([hid, c0, c1]); });
      txt = bi(() => {
        const what = lg(["手中星象各自随机变为另两种之一", "手中星象星阶 +1", "手中星象星阶 −1"][lid], ["Asterisms in hand randomly transform into another type", "Asterisms in hand +1 Brilliance", "Asterisms in hand −1 Brilliance"][lid]);
        let x = `${head()}${lg("：", ": ")}${what}` + (chg.length ? "" : lg("（无变化）", " (no change)"));
        if (chg.length) x += lg("：", ": ") + chg.map(([, a, b]) => `${sn(a)}→${sn(b)}`).join(lg("、", ", "));
        return x;
      });
    } else if (lid === 3) {
      const o = []; for (let j = 0; j < si[O_PN]; j++) o.push(si[O_PO + j]);
      txt = bi(() => `${head()}${lg("：发现 3 张法则 —— ", ": discover 3 Runes — ")}` + o.map(ln).join(" / "));
    } else if (lid === 4) {
      const nNew = si[O_HN] - hn0, forms = [];
      for (let q = 0; q < nNew; q++) forms.push(si[O_HC + hn0 + q]);
      for (let q = 0; q < nNew; q++) this.hand.push(this.nid());
      txt = bi(() => {
        let x = `${head()}${lg("：抽 2 张星象", ": draw 2 Asterisms")}` + (forms.length ? lg("：", ": ") + forms.map(sn).join(lg("、", ", ")) : "");
        if (nNew < 2) x += lg(`（手牌已满，${2 - nNew} 张进弃牌堆）`, ` (hand full, ${2 - nNew} to the Discard Pile)`);
        return x;
      });
    } else if (lid === 5) {
      const base = ln0 - 1, nNew = si[O_LN] - base, names = [];
      for (let q = 0; q < nNew; q++) names.push(si[O_LI + base + q]);
      for (let q = 0; q < nNew; q++) this.laws.push(this.nid());
      txt = bi(() => {
        let x = `${head()}${lg("：获得 3 张法则", ": gain 3 Runes")}` + (names.length ? lg("：", ": ") + names.map(ln).join(lg("、", ", ")) : "");
        if (nNew < 3) x += lg(`（手牌已满，${3 - nNew} 张消失）`, ` (hand full, ${3 - nNew} lost)`);
        return x;
      });
    } else {
      this.hand = []; this.laws = []; this.track = [];
      for (let i = 0; i < si[O_HN]; i++) this.hand.push(this.nid());
      for (let i = 0; i < si[O_LN]; i++) this.laws.push(this.nid());
      txt = bi(() => {
        let x = `${head()}${lg("：清空星轨与手牌，重新抽 4 张星象、获得 2 张法则", ": clear the Orbits and hand, draw 4 Asterisms, gain 2 Runes")}`;
        if (locked) x += lg("（已放入星轨的卡效已白嫖）", " (effects of cards already in the Orbits are kept)");
        return x;
      });
    }
    return this.emit("law", txt, null, { played: [lawid, lid, cost], chg });
  }
  lawById(lid, paid) {
    const si = this.si;
    for (let li = 0; li < si[O_LN]; li++) if (si[O_LI + li] === lid && ((si[O_LC + li] > 0) === paid) && si[O_I] >= si[O_LC + li]) return this.lawIdx(li);
    throw err("找不到法则", "Rune not found");
  }
  pick(j) {
    const { si, cfg } = this;
    const lid = si[O_PO + j], ln0 = si[O_LN];
    pickOption(si, cfg, j);
    const got = si[O_LN] > ln0;
    if (got) this.laws.push(this.nid());
    const txt = bi(() => got ? lg(`昭示：选取「${ln(lid)}」，本回合消耗变为 0`, `Herald: picked ${ln(lid)}, its cost becomes 0 this turn`) : lg(`昭示：选取「${ln(lid)}」，但手牌已满，无法获得`, `Herald: picked ${ln(lid)}, but the hand is full so it is lost`));
    return this.emit("pick", txt, null, { picked: lid });
  }
  finishPlan(plan) {
    while (this.si[O_TN] < 3 && !this.done()) {
      this.placeForm(plan[this.si[O_TN]]);
      if (this.si[O_TN] === 0) break;
    }
  }
  legal() {
    const si = this.si;
    if (this.done()) return { place: [], law: [], pick: [] };
    if (si[O_PN] > 0) { const p = []; for (let j = 0; j < si[O_PN]; j++) p.push(j); return { place: [], law: [], pick: p }; }
    const place = [], law = [];
    for (let i = 0; i < si[O_HN]; i++) place.push(this.hand[i]);
    for (let i = 0; i < si[O_LN]; i++) if (si[O_I] >= si[O_LC + i]) law.push(this.laws[i]);
    return { place, law, pick: [] };
  }
  /** 手动操作：kind = place / law（arg 为卡牌 id）或 pick（arg 为发现项下标）。随机数按（种子, 步数）重设，保证可复现。 */
  act(kind, arg) {
    if (this.done()) throw err("本局已结束", "This game is over");
    const lg = this.legal();
    const ok = (kind === "place" && lg.place.includes(arg)) || (kind === "law" && lg.law.includes(arg)) || (kind === "pick" && lg.pick.includes(arg));
    if (!ok) throw err("当前不能这样操作", "That move is not allowed now");
    this.nact++;
    seedRng(this.seed * 1000003 + this.nact * 7919 + 17);
    const n0 = this.frames.length;
    if (kind === "place") this.placeIdx(this.hand.indexOf(arg));
    else if (kind === "law") this.lawIdx(this.laws.indexOf(arg));
    else this.pick(arg);
    const out = { frames: this.frames.slice(n0), legal: this.legal(), done: this.done() };
    if (out.done) out.summary = this.summary();
    return out;
  }
  summary() {
    const st = this.st, sf = this.sf;
    return { policy: "you", seed: this.seed, total: r3(sf[F_R]), turnR: Array.from(st.subarray(ST_TURNR, ST_TURNR + 6)).map(x => Math.round(x * 100) / 100),
      laws: st[ST_NLAW], lawcnt: Array.from(st.subarray(ST_LAWCNT, ST_LAWCNT + 7)), n3Z: st[ST_N3Z], n3Y: st[ST_N3Y], n3H: st[ST_N3H], Ileft: this.si[O_I] };
  }
  result(policy) {
    const r = this.summary(); r.policy = policy; r.decs = this.decs; r.frames = this.frames; return r;
  }
}

// =========================================================================
// 策略驱动（生成器：yield 处可让出主线程；runSync 同步跑完）
// =========================================================================
function planLabel(plan, tn) { const o = []; for (let i = tn; i < 3; i++) o.push(plan[i]); return bi(() => lg("补满星轨：", "Fill the Orbits: ") + o.map(sn).join(" → ")); }
function lawLabel(lid, cost, out) {
  const locks = [out[0], out[1]].filter(x => x >= 0);
  return bi(() => {
    let lab = lg(`打出「${ln(lid)}」` + (cost ? "" : "（0 消耗）"), `Play ${ln(lid)}` + (cost ? "" : " (cost 0)"));
    if (locks.length) lab += lg("，先把 " + locks.map(sn).join("、") + " 放进星轨", ", first placing " + locks.map(sn).join(", ") + " in the Orbits");
    return lab;
  });
}
function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

function playParam(game, theta, K = 12) {
  const { si, sf, cfg } = game;
  const plan0 = new Int32Array(3), out = new Int32Array(2), outb = new Int32Array(2);
  let guard = 0;
  while (!game.done() && guard < 400) {
    guard++;
    const t = si[O_T], hn = si[O_HN], tn = si[O_TN];
    const V0 = bestCompletion(si.subarray(O_HC, O_HC + 16), hn, si.subarray(O_TC, O_TC + 3), tn, sf[F_B], t, cfg, theta, plan0);
    if (si[O_PN] > 0) {
      const cands = [];
      for (let j = 0; j < si[O_PN]; j++) {
        const lid = si[O_PO + j];
        cands.push([bi(() => lg(`选取「${ln(lid)}」`, `Pick ${ln(lid)}`)), lawValue2(lid, 0, si, sf[F_B], t, cfg, theta, K, V0, out)]);
      }
      const bj = argmax(cands.map(c => c[1]));
      game.decision(["昭示：从 3 张法则中选 1 张", "Herald: choose 1 of 3 Runes"], "评分", cands, bj);
      game.pick(bj);
      continue;
    }
    const cands = [], acts = [];
    if (V0 > -1e17) { cands.push([planLabel(plan0, tn), V0]); acts.push(["finish", null, null, Int32Array.from(plan0)]); }
    let best = V0, bestl = -1; outb.fill(-1);
    const seen = new Set();
    for (let li = 0; li < si[O_LN]; li++) {
      const lid = si[O_LI + li], cost = si[O_LC + li], key = lid * 100 + cost;
      if (si[O_I] < cost || seen.has(key)) continue;
      seen.add(key);
      const v = lawValue2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out);
      cands.push([lawLabel(lid, cost, out), v]); acts.push(["law", li]);
      if (v > best + 1e-9) { best = v; bestl = li; outb[0] = out[0]; outb[1] = out[1]; }
    }
    if (!cands.length) break;
    let ci;
    if (bestl >= 0) ci = acts.findIndex(a => a[0] === "law" && a[1] === bestl);
    else if (V0 > -1e17) ci = 0;
    else break;
    game.decision([`第 ${t} 回合：补满星轨，还是先打法则？`, `Turn ${t}: fill the Orbits, or play a Rune first?`], "评分", cands, ci);
    if (acts[ci][0] === "law") {
      for (let q = 0; q < 2; q++) if (outb[q] >= 0) game.placeForm(outb[q]);
      game.lawIdx(bestl);
    } else game.finishPlan(plan0);
  }
  return game;
}

/** 滚动前瞻专家在当前状态的评估（生成器，可 yield 让出线程）。返回 [标题, 单位, 候选[[标签,数值]], 动作表, 补满方案, 最优下标]。 */
function* expertEvalG(game, theta, K, R, sb) {
  const { si, sf, cfg } = game;
  const plan0 = new Int32Array(3), out = new Int32Array(2);
  const t = si[O_T], hn = si[O_HN], tn = si[O_TN];
  const V0 = bestCompletion(si.subarray(O_HC, O_HC + 16), hn, si.subarray(O_TC, O_TC + 3), tn, sf[F_B], t, cfg, theta, plan0);
  const cands = [], acts = [];
  if (si[O_PN] > 0) {
    for (let j = 0; j < si[O_PN]; j++) {
      const v = actionValue(si, sf, theta, cfg, K, R, sb, KIND_OPTION, j, -1, -1, 0, 0, 0);
      { const lid2 = si[O_PO + j]; cands.push([bi(() => lg(`选取「${ln(lid2)}」`, `Pick ${ln(lid2)}`)), v]); } acts.push(["opt", j, -1, -1]); yield;
    }
    return [["昭示：从 3 张法则中选 1 张", "Herald: choose 1 of 3 Runes"], "期望总倍率", cands, acts, plan0, argmax(cands.map(c => c[1]))];
  }
  if (V0 > -1e17) {
    const v = actionValue(si, sf, theta, cfg, K, R, sb, KIND_FINISH, 0, -1, -1, plan0[0], plan0[1], plan0[2]);
    cands.push([planLabel(plan0, tn), v]); acts.push(["finish", null, -1, -1]); yield;
  }
  const seen = new Set();
  for (let li = 0; li < si[O_LN]; li++) {
    const lid = si[O_LI + li], cost = si[O_LC + li], key = lid * 100 + cost;
    if (si[O_I] < cost || seen.has(key)) continue;
    seen.add(key);
    lawValue2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out);
    const l1 = out[0], l2 = out[1];
    let v = actionValue(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, l1, l2, 0, 0, 0);
    cands.push([lawLabel(lid, cost, [l1, l2]), v]); acts.push(["law", li, l1, l2]); yield;
    if (l1 >= 0 || l2 >= 0) {
      v = actionValue(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, -1, -1, 0, 0, 0);
      { const lb = lawLabel(lid, cost, [-1, -1]); cands.push([[lb[0] + "（不提前放牌）", lb[1] + " (no cards placed first)"], v]); } acts.push(["law", li, -1, -1]); yield;
    }
  }
  return [[`第 ${t} 回合：滚动前瞻（每个候选各模拟 ${R} 局）`, `Turn ${t}: rollout lookahead (${R} simulated games per candidate)`], "期望总倍率", cands, acts, plan0, cands.length ? argmax(cands.map(c => c[1])) : -1];
}
function* playExpertG(game, theta, K = 12, R = 200) {
  let dec = 0;
  const gseed = game.seed;
  while (!game.done() && dec < 400) {
    dec++;
    const sb = (gseed * 7919 + dec * 104729) >>> 0;
    const [title, unit, cands, acts, plan0, ci] = yield* expertEvalG(game, theta, K, R, sb);
    if (!cands.length) break;
    game.decision(title, unit, cands, ci);
    const [kind, a, l1, l2] = acts[ci];
    if (kind === "opt") game.pick(a);
    else if (kind === "finish") game.finishPlan(plan0);
    else { if (l1 >= 0) game.placeForm(l1); if (l2 >= 0) game.placeForm(l2); game.lawIdx(a); }
    seedRng(gseed * 1000003 + dec * 7919 + 17);
    yield;
  }
  return game;
}
function playPpo(game, model, topk = 5) {
  const { si, sf, cfg } = game;
  const obs = new Float64Array(OBS_DIM), mask = new Array(NUM_ACTIONS).fill(false);
  let guard = 0;
  while (!game.done() && guard < 400) {
    guard++;
    buildObsMask(si, sf, cfg, obs, mask);
    if (!mask.some(Boolean)) break;
    const p = ppoProbs(model, obs, mask);
    let order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, topk).filter(a => mask[a] && p[a] > 1e-4);
    if (!order.length) order = [argmax(p)];
    const a = order[0];
    game.decision([`第 ${si[O_T]} 回合：网络给出的动作概率`, `Turn ${si[O_T]}: action probabilities from the network`], "概率 %", order.map(x => [ACTION_NAMES[x], p[x] * 100]), 0);
    if (a < A_LAW0) game.placeForm(a);
    else if (a < A_OPT0) { const k = a - A_LAW0; game.lawById(k % 7, k < 7); }
    else { const lid = a - A_OPT0; let j = 0; for (; j < si[O_PN]; j++) if (si[O_PO + j] === lid) break; game.pick(j); }
  }
  return game;
}

// ---- 运行生成器：同步 / 异步（每 ~30ms 让出一次主线程） ----
function runSync(gen) { let r = gen.next(); while (!r.done) r = gen.next(); return r.value; }
async function runAsync(gen, onProgress) {
  let last = Date.now(), r = gen.next(), v;
  while (!r.done) {
    if (typeof r.value === "number") v = r.value;
    if (Date.now() - last > 30) { if (onProgress) onProgress(v); await new Promise(res => setTimeout(res, 0)); last = Date.now(); }
    r = gen.next();
  }
  return r.value;
}

// =========================================================================
// 顶层接口
// =========================================================================
const P = root.TULU_PARAMS;
const CFG = Float64Array.from(P.cfg), THETA = Float64Array.from(P.theta);
let MODEL = null;
function getModel() {
  if (MODEL) return MODEL;
  const m = root.TULU_MODEL;
  if (!m) return null;
  MODEL = { hidden: m.hidden, W1: Float32Array.from(m.W1), b1: Float32Array.from(m.b1), W2: Float32Array.from(m.W2), b2: Float32Array.from(m.b2),
    Wp: Float32Array.from(m.Wp), bp: Float32Array.from(m.bp) };
  return MODEL;
}
const BINS = [700.0, 1700.0, 40];

function newGame(seed) { return new Game(CFG, seed); }

/** 某策略在某种子下打一整局（生成器）。 */
function* runPolicyG(policy, seed, opts = {}) {
  const g = new Game(CFG, seed);
  if (policy === "param") playParam(g, THETA);
  else if (policy === "expert") yield* playExpertG(g, THETA, 12, opts.R || 200);
  else { const m = getModel(); if (!m) throw err("PPO 权重不可用", "PPO weights unavailable"); playPpo(g, m); }
  const ep = g.result(policy); ep.id = `${policy}-${seed}`;
  return ep;
}
/** 专家建议（生成器）：返回 {title, unit, cands:[[标签, 数值, 是否最优]]}。 */
function* hintG(game, R = 100) {
  if (game.done()) throw err("本局已结束", "This game is over");
  seedRng(game.seed * 7 + game.nact * 131 + 5);
  const [title, unit, cands, , , ci] = yield* expertEvalG(game, THETA, 12, R, (game.seed * 7919 + game.nact * 104729) >>> 0);
  return { title: title[0], titleEn: title[1], unit, cands: cands.map(([lb, v], i) => [lb[0], Math.round(v * 100) / 100, i === ci ? 1 : 0, lb[1]]) };
}
/** 同开局（相同初始手牌与法则）、之后随机事件独立，让程序重复打 n 局（生成器，yield 进度）。 */
function* policyDistG(policy, seed, n, opts = {}) {
  const base = new Game(CFG, seed);
  const totals = [];
  for (let k = 0; k < n; k++) {
    if (policy === "param") {
      const si = Int32Array.from(base.si), sf = Float64Array.from(base.sf), st = new Float64Array(ST_LEN);
      seedRng(seed * 100003 + k + 1);
      playUntilDone(si, sf, THETA, CFG, 12, 0, st);
      totals.push(sf[F_R]);
    } else {
      const g = new Game(CFG, seed);
      g.seed = seed * 100003 + k + 1; seedRng(g.seed);
      if (policy === "expert") yield* playExpertG(g, THETA, 12, opts.R || 60);
      else playPpo(g, getModel());
      totals.push(g.sf[F_R]);
    }
    yield k + 1;
  }
  return totals;
}
function distStats(tot, my) {
  const s = tot.slice().sort((a, b) => a - b), n = s.length;
  const mean = s.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n);
  const q = p => { const x = (n - 1) * p / 100, lo = Math.floor(x), hi = Math.ceil(x); return s[lo] + (s[hi] - s[lo]) * (x - lo); };
  const [lo, hi, nb] = BINS, hist = new Array(nb).fill(0);
  tot.forEach(v => { let b = Math.floor((Math.min(Math.max(v, lo), hi - 1e-6) - lo) / (hi - lo) * nb); hist[b]++; });
  const out = { n, mean, sd, p05: q(5), p50: q(50), p95: q(95), hist, bins: BINS };
  if (my !== undefined) out.pct = tot.filter(v => v < my).length / n * 100;
  return out;
}
/** 与程序解对比（生成器）：同种子单局回放 + 同开局重复 n 局的分布。 */
function* compareG(seed, policy, n, my, opts = {}) {
  const ep = yield* runPolicyG(policy, seed, { R: opts.RSingle || 100 });
  const tot = yield* policyDistG(policy, seed, n, { R: opts.RDist || 60 });
  return Object.assign({ policy, ep }, distStats(tot, my));
}

root.TuluEngine = { Game, newGame, runPolicyG, hintG, compareG, policyDistG, distStats, runSync, runAsync, playParam, playUntilDone, playExpertG, playPpo,
  seedRng, initGame, CFG, THETA, getModel, STAR_NAMES, LAW_NAMES, ST_TOTAL, ST_LEN, F_R, SI_LEN, SF_LEN, Int32Array, mode: { random: 1, param: 0 } };
})(typeof globalThis !== "undefined" ? globalThis : (typeof window !== "undefined" ? window : this));
