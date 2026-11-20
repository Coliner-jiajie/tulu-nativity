"""可视化回放录制：让策略打一局，把每个原子动作与结算拆成「帧」，供网页逐帧回放。

每帧是一个完整快照（手牌、法则、星轨、抽牌堆/弃牌堆计数、铭文、触腕伤害、累计倍率）加上事件描述。
卡牌带唯一 id，前端据此做飞入/飞出动画。三种策略：
  * param  ：方法 A，参数化打分策略（CEM 最优 θ）
  * expert ：滚动前瞻专家（目前最强）
  * ppo    ：方法 B，PPO 神经网络（贪心）
帧类型 ev：start（回合开始抽牌）、place（放入星轨）、law（打出法则）、pick（昭示选取）、
          tide（星潮奔涌）、end（回合结束：弃牌 + 余波）、over（全部结束）。
"""
import numpy as np
from numba import njit

from tulu.env.core import *
from tulu.param.policy import best_completion, law_value2
from tulu.param.rollout import KIND_FINISH, KIND_LAW, KIND_OPTION, action_value
from tulu.common.names import STAR_NAMES, LAW_NAMES, action_names


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


def _cnt(lst, n=NT):
    out = [0] * n
    for x in lst:
        out[x] += 1
    return out


class Game:
    def __init__(self, cfg, seed):
        self.cfg = np.asarray(cfg, dtype=np.float64)
        self.seed = int(seed)
        self.si = np.zeros(SI_LEN, dtype=np.int64)
        self.sf = np.zeros(SF_LEN)
        self.st = np.zeros(ST_LEN)
        seed_rng(self.seed)
        init_game(self.si, self.sf, self.cfg)
        self._n = 0
        self.hand, self.laws, self.track = [], [], []
        self.frames, self.decs = [], []
        self.cur_d, self.d_new = None, False
        self.new_turn_ids()
        self.emit("start", self._start_text(first=True))

    # ---- 基础工具 --------------------------------------------------------
    def nid(self):
        self._n += 1
        return self._n

    def done(self):
        return bool(self.si[O_DONE])

    def hand_list(self):
        si = self.si
        return [[self.hand[i], int(si[O_HC + i]), int(si[O_HO + i])] for i in range(int(si[O_HN]))]

    def laws_list(self):
        si = self.si
        return [[self.laws[i], int(si[O_LI + i]), int(si[O_LC + i])] for i in range(int(si[O_LN]))]

    def track_list(self):
        si = self.si
        return [[self.track[i], int(si[O_TC + i]), int(si[O_TO + i])] for i in range(int(si[O_TN]))]

    def snap(self, hand=None, laws=None, track=None, I=None, B=None, R=None, dp=None, dc=None, t=None, opts=None):
        si, sf = self.si, self.sf
        return {
            "t": int(min(si[O_T], self.cfg[C_N])) if t is None else t,
            "h": self.hand_list() if hand is None else hand,
            "l": self.laws_list() if laws is None else laws,
            "k": self.track_list() if track is None else track,
            "o": [int(si[O_PO + j]) for j in range(int(si[O_PN]))] if opts is None else opts,
            "I": int(si[O_I]) if I is None else int(I),
            "B": round(float(sf[F_B] if B is None else B), 4),
            "R": round(float(sf[F_R] if R is None else R), 3),
            "dp": [int(x) for x in si[O_DP:O_DP + NT]] if dp is None else dp,
            "dc": [int(x) for x in si[O_DC:O_DC + NT]] if dc is None else dc,
        }

    def emit(self, ev, txt, snap=None, **kw):
        f = snap if snap is not None else self.snap()
        f["ev"] = ev
        f["tx"] = txt
        if self.cur_d is not None:
            f["d"] = self.cur_d
            if self.d_new:
                f["df"] = 1
                self.d_new = False
        f.update(kw)
        self.frames.append(f)
        return f

    def decision(self, title, unit, cands, chosen):
        """登记一次决策：cands = [(标签, 数值), ...]，chosen 为选中的下标。"""
        self.decs.append({"t": int(self.si[O_T]), "ti": title, "u": unit,
                          "c": [[lb, round(float(v), 3), 1 if i == chosen else 0] for i, (lb, v) in enumerate(cands)]})
        self.cur_d = len(self.decs) - 1
        self.d_new = True

    def new_turn_ids(self):
        si = self.si
        self.hand = [self.nid() for _ in range(int(si[O_HN]))]
        self.laws = [self.nid() for _ in range(int(si[O_LN]))]
        self.track = []

    def _start_text(self, first=False):
        t = int(self.si[O_T])
        s = f"第 {t} 回合开始：抽 4 张星象，随机获得 2 张法则"
        if t > 1:
            s += f"；触腕伤害自然增长到 {self.sf[F_B]:.3f}"
        return s

    # ---- 放入星轨 --------------------------------------------------------
    def place_form(self, cur):
        si = self.si
        for i in range(int(si[O_HN])):
            if si[O_HC + i] == cur:
                return self.place_idx(i)
        raise RuntimeError(f"手中没有形态为 {STAR_NAMES[cur]} 的星象")

    def place_idx(self, hi):
        si, sf, cfg, st = self.si, self.sf, self.cfg, self.st
        cur, orig, cid = int(si[O_HC + hi]), int(si[O_HO + hi]), self.hand[hi]
        j = int(si[O_TN]) + 1
        k = cur // 5
        eff = float(f_of(k, cfg))
        dmg = eff * j
        dB = float(cfg[C_CARDB] * cfg[C_RHO] * cfg[C_M] * j) if k == 2 else 0.0
        hand0, laws0, track0 = self.hand_list(), self.laws_list(), self.track_list()
        hand_after = [h for h in hand0 if h[0] != cid]
        track_after = track0 + [[cid, cur, orig]]
        R0, B0, I0, t = float(sf[F_R]), float(sf[F_B]), int(si[O_I]), int(si[O_T])
        dp0 = [int(x) for x in si[O_DP:O_DP + NT]]
        dc0 = [int(x) for x in si[O_DC:O_DC + NT]]
        self.hand[hi] = self.hand[-1]
        self.hand.pop()
        self.track.append(cid)
        txt = f"{STAR_NAMES[cur]} 放入星轨第 {j} 位，卡效触发 {j} 次：{eff:g} × {j} = {dmg:.1f}"
        if dB > 0:
            txt += f"；触腕伤害 +{dB:.4f}"
        if j < 3:
            place_star(si, sf, cfg, hi, st)
            return self.emit("place", txt, cid=cid, slot=j, dmg=round(dmg, 3), dB=round(dB, 4), kind=k)
        # ---- 第 3 张：卡效 → 星潮奔涌 → 回合结束 ----
        cs = [track_after[i][1] for i in range(3)]
        B1 = B0 + dB
        tdmg, nZ, nH, nY, hits = tide_calc(cs[0], cs[1], cs[2], B1, cfg)
        info = self.tide_info(cs, B1, float(tdmg), int(nZ), int(nH), int(nY), int(hits))
        place_star(si, sf, cfg, hi, st)
        R1 = R0 + dmg
        self.emit("place", txt, snap=self.snap(hand=hand_after, laws=laws0, track=track_after, I=I0, B=B1, R=R1,
                                               dp=dp0, dc=dc0, t=t, opts=[]),
                  cid=cid, slot=j, dmg=round(dmg, 3), dB=round(dB, 4), kind=k)
        turn_r = float(st[ST_TURNR + t - 1])
        self.emit("tide", info["txt"], snap=self.snap(hand=hand_after, laws=laws0, track=track_after, I=I0, B=B1,
                                                      R=R1 + tdmg, dp=dp0, dc=dc0, t=t, opts=[]),
                  tide=info, tr=round(turn_r, 3))
        dcE = list(dc0)
        for h in hand_after:
            dcE[h[2]] += 1
        for q in track_after:
            dcE[q[2]] += 1
        gain = min(int(cfg[C_IMAX]), I0 + int(cfg[C_AFTER]) * len(laws0)) - I0
        B2 = B1 + float(cfg[C_TIDEB] * cfg[C_RHO] * cfg[C_M] * nY)
        etxt = f"第 {t} 回合结束：本回合共 {turn_r:.1f}；弃置手牌与星轨"
        if len(laws0):
            etxt += f"；{len(laws0)} 张未用法则消散，余波使铭文 +{gain}"
        self.emit("end", etxt, snap=self.snap(hand=[], laws=[], track=[], I=I0 + gain, B=B2, R=float(sf[F_R]),
                                              dp=dp0, dc=dcE, t=t, opts=[]), gain=gain, nlaw=len(laws0))
        if self.done():
            self.cur_d = None
            self.emit("over", f"诞临结束：6 回合合计 {sf[F_R]:.1f}（另有「忘却前夜」+80）")
        else:
            self.new_turn_ids()
            self.cur_d = None
            self.emit("start", self._start_text())

    def tide_info(self, cs, B, dmg, nZ, nH, nY, hits):
        cfg = self.cfg
        ks = [c // 5 for c in cs]
        rs = [c % 5 + 1 for c in cs]
        if rs[0] == rs[1] == rs[2]:
            a, an = 6, "同星阶 +6"
        elif sorted(rs) == [min(rs), min(rs) + 1, min(rs) + 2]:
            a, an = 3, "三连号 +3"
        else:
            a, an = 0, ""
        if ks[0] == ks[1] == ks[2]:
            b, bn = 6, "同种类 +6"
        elif len(set(ks)) == 3:
            b, bn = 3, "三种各一 +3"
        else:
            b, bn = 0, ""
        per = float(cfg[C_BASEHIT] + cfg[C_PERZ] * nZ)
        bonus = float(cfg[C_KAPPA] * B) if nZ == 3 else 0.0
        pat = "".join("灾垣蕴"[k] for k in ks)
        txt = f"星潮奔涌：{hits} 次 × {per + bonus:.2f} = {dmg:.1f}"
        if nY == 3:
            txt += f"（蕴蕴蕴 ×{cfg[C_Y3]:g}）"
        return {"hits": hits, "a": a, "an": an, "b": b, "bn": bn, "per": round(per, 3), "bonus": round(bonus, 3),
                "dmg": round(dmg, 3), "nZ": nZ, "nH": nH, "nY": nY, "pat": pat, "cs": cs, "txt": txt,
                "y3": float(cfg[C_Y3]) if nY == 3 else 1.0, "B": round(B, 4)}

    # ---- 法则 ------------------------------------------------------------
    def law_idx(self, li):
        si, sf, cfg, st = self.si, self.sf, self.cfg, self.st
        lid, cost, lawid = int(si[O_LI + li]), int(si[O_LC + li]), self.laws[li]
        hand0 = self.hand_list()
        hn0, ln0 = int(si[O_HN]), int(si[O_LN])
        locked = int(si[O_TN]) > 0
        self.laws[li] = self.laws[-1]
        self.laws.pop()
        play_law(si, sf, cfg, li, st)
        chg, txt = [], ""
        name = LAW_NAMES[lid]
        head = f"打出「{name}」" + (f"（消耗 {cost} 层铭文）" if cost else "（0 消耗）")
        if lid in (0, 1, 2):
            for i, (hid, c0, _) in enumerate(hand0):
                c1 = int(si[O_HC + i])
                if c1 != c0:
                    chg.append([hid, c0, c1])
            what = {0: "手中星象各自随机变为另两种之一", 1: "手中星象星阶 +1", 2: "手中星象星阶 −1"}[lid]
            txt = f"{head}：{what}" + ("" if chg else "（无变化）")
            if chg:
                txt += "：" + "、".join(f"{STAR_NAMES[a]}→{STAR_NAMES[b]}" for _, a, b in chg)
        elif lid == 3:
            opts = [LAW_NAMES[int(si[O_PO + j])] for j in range(int(si[O_PN]))]
            txt = f"{head}：发现 3 张法则 —— " + " / ".join(opts)
        elif lid == 4:
            n_new = int(si[O_HN]) - hn0
            forms = [STAR_NAMES[int(si[O_HC + hn0 + q])] for q in range(n_new)]
            self.hand += [self.nid() for _ in range(n_new)]
            txt = f"{head}：抽 2 张星象" + (f"：{'、'.join(forms)}" if forms else "")
            if n_new < 2:
                txt += f"（手牌已满，{2 - n_new} 张进弃牌堆）"
        elif lid == 5:
            base = ln0 - 1
            n_new = int(si[O_LN]) - base
            names = [LAW_NAMES[int(si[O_LI + base + q])] for q in range(n_new)]
            self.laws += [self.nid() for _ in range(n_new)]
            txt = f"{head}：获得 3 张法则" + (f"：{'、'.join(names)}" if names else "")
            if n_new < 3:
                txt += f"（手牌已满，{3 - n_new} 张消失）"
        else:
            self.hand = [self.nid() for _ in range(int(si[O_HN]))]
            self.laws = [self.nid() for _ in range(int(si[O_LN]))]
            self.track = []
            txt = f"{head}：清空星轨与手牌，重新抽 4 张星象、获得 2 张法则"
            if locked:
                txt += "（已放入星轨的卡效已白嫖）"
        return self.emit("law", txt, played=[lawid, lid, cost], chg=chg)

    # ---- 手动操作（按卡牌 id） ----
    def legal(self):
        si = self.si
        if self.done():
            return {"place": [], "law": [], "pick": []}
        if si[O_PN] > 0:
            return {"place": [], "law": [], "pick": list(range(int(si[O_PN])))}
        return {"place": [self.hand[i] for i in range(int(si[O_HN]))],
                "law": [self.laws[i] for i in range(int(si[O_LN])) if si[O_I] >= si[O_LC + i]],
                "pick": []}

    def act(self, kind, arg):
        """kind: place / law（arg 为卡牌 id）；pick（arg 为发现项下标）。非法操作抛 ValueError。"""
        if self.done():
            raise ValueError("本局已结束")
        lg = self.legal()
        if kind == "place" and arg in lg["place"]:
            return self.place_idx(self.hand.index(arg))
        if kind == "law" and arg in lg["law"]:
            return self.law_idx(self.laws.index(arg))
        if kind == "pick" and arg in lg["pick"]:
            return self.pick(arg)
        raise ValueError("当前不能这样操作")

    def law_by_id(self, lid, paid):
        si = self.si
        for li in range(int(si[O_LN])):
            if si[O_LI + li] == lid and ((si[O_LC + li] > 0) == paid) and si[O_I] >= si[O_LC + li]:
                return self.law_idx(li)
        raise RuntimeError("找不到法则")

    def pick(self, j):
        si, cfg = self.si, self.cfg
        lid = int(si[O_PO + j])
        ln0 = int(si[O_LN])
        pick_option(si, cfg, j)
        got = int(si[O_LN]) > ln0
        if got:
            self.laws.append(self.nid())
        txt = f"昭示：选取「{LAW_NAMES[lid]}」，本回合消耗变为 0" if got else f"昭示：选取「{LAW_NAMES[lid]}」，但手牌已满，无法获得"
        return self.emit("pick", txt, picked=lid)

    def finish_plan(self, plan):
        """按计划把星轨补满。"""
        while int(self.si[O_TN]) < 3 and not self.done():
            self.place_form(int(plan[int(self.si[O_TN])]))
            if int(self.si[O_TN]) == 0:
                break

    def result(self, policy):
        sf, st = self.sf, self.st
        return {
            "policy": policy, "seed": self.seed, "total": round(float(sf[F_R]), 3),
            "turnR": [round(float(x), 2) for x in st[ST_TURNR:ST_TURNR + 6]],
            "laws": int(st[ST_NLAW]), "lawcnt": [int(x) for x in st[ST_LAWCNT:ST_LAWCNT + 7]],
            "n3Z": int(st[ST_N3Z]), "n3Y": int(st[ST_N3Y]), "n3H": int(st[ST_N3H]),
            "Ileft": int(self.si[O_I]),
            "decs": self.decs, "frames": self.frames,
        }


# =========================================================================
# 策略驱动
# =========================================================================
def _law_label(lid, cost, out):
    lab = f"打出「{LAW_NAMES[lid]}」" + ("" if cost else "（0 消耗）")
    locks = [STAR_NAMES[int(x)] for x in out if x >= 0]
    if locks:
        lab += "，先把 " + "、".join(locks) + " 放进星轨"
    return lab


def _plan_label(plan, tn):
    return "补满星轨：" + " → ".join(STAR_NAMES[int(plan[i])] for i in range(tn, 3))


def play_param(game, theta, K=12):
    """方法 A：与 tulu.param.policy.play_until_done(mode=0) 同一逻辑，但逐步录制并记录各候选的评分。"""
    si, sf, cfg = game.si, game.sf, game.cfg
    plan0 = np.zeros(3, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    outb = np.zeros(2, dtype=np.int64)
    guard = 0
    while not game.done() and guard < 400:
        guard += 1
        t, hn, tn = int(si[O_T]), int(si[O_HN]), int(si[O_TN])
        V0 = float(best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0))
        if si[O_PN] > 0:
            cands = []
            for j in range(int(si[O_PN])):
                lid = int(si[O_PO + j])
                v = float(law_value2(lid, 0, si, sf[F_B], t, cfg, theta, K, V0, out))
                cands.append((f"选取「{LAW_NAMES[lid]}」", v))
            bj = int(np.argmax([v for _, v in cands]))
            game.decision("昭示：从 3 张法则中选 1 张", "评分", cands, bj)
            game.pick(bj)
            continue
        cands = []
        acts = []
        if V0 > -1e17:
            cands.append((_plan_label(plan0, tn), V0))
            acts.append(("finish", None, None, plan0.copy()))
        best, bestl = V0, -1
        outb[:] = -1
        seen = set()
        for li in range(int(si[O_LN])):
            lid, cost = int(si[O_LI + li]), int(si[O_LC + li])
            if si[O_I] < cost or (lid, cost) in seen:
                continue
            seen.add((lid, cost))
            v = float(law_value2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out))
            cands.append((_law_label(lid, cost, out), v))
            acts.append(("law", li, out.copy(), None))
            if v > best + 1e-9:
                best, bestl = v, li
                outb[:] = out
        if not cands:
            break
        # 选中项：与原策略一致（法则需严格优于补满星轨）
        if bestl >= 0:
            ci = next(i for i, a in enumerate(acts) if a[0] == "law" and a[1] == bestl)
        elif V0 > -1e17:
            ci = 0
        else:
            break
        game.decision(f"第 {t} 回合：补满星轨，还是先打法则？", "评分", cands, ci)
        if acts[ci][0] == "law":
            for q in range(2):
                if outb[q] >= 0:
                    game.place_form(int(outb[q]))
            game.law_idx(bestl)
        else:
            game.finish_plan(plan0)
    return game


def expert_eval(game, theta, K, R, sb):
    """滚动前瞻专家在当前状态的评估：返回 (标题, 单位, 候选[(标签, 数值)], 动作表, 补满方案, 最优下标)。
    动作表项 (kind, a, l1, l2)：kind 为 'opt' / 'finish' / 'law'。当前状态必须是一个全新的决策点（无半途计划）。"""
    si, sf, cfg = game.si, game.sf, game.cfg
    plan0 = np.zeros(3, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    t, hn, tn = int(si[O_T]), int(si[O_HN]), int(si[O_TN])
    V0 = float(best_completion(si[O_HC:O_HC + 16], hn, si[O_TC:O_TC + 3], tn, sf[F_B], t, cfg, theta, plan0))
    cands, acts = [], []
    if si[O_PN] > 0:
        for j in range(int(si[O_PN])):
            v = float(action_value(si, sf, theta, cfg, K, R, sb, KIND_OPTION, j, -1, -1, 0, 0, 0))
            cands.append((f"选取「{LAW_NAMES[int(si[O_PO + j])]}」", v))
            acts.append(("opt", j, -1, -1))
        return "昭示：从 3 张法则中选 1 张", "期望总倍率", cands, acts, plan0, int(np.argmax([v for _, v in cands]))
    if V0 > -1e17:
        v = float(action_value(si, sf, theta, cfg, K, R, sb, KIND_FINISH, 0, -1, -1, plan0[0], plan0[1], plan0[2]))
        cands.append((_plan_label(plan0, tn), v))
        acts.append(("finish", None, -1, -1))
    seen = set()
    for li in range(int(si[O_LN])):
        lid, cost = int(si[O_LI + li]), int(si[O_LC + li])
        if si[O_I] < cost or (lid, cost) in seen:
            continue
        seen.add((lid, cost))
        law_value2(lid, cost, si, sf[F_B], t, cfg, theta, K, V0, out)
        l1, l2 = int(out[0]), int(out[1])
        v = float(action_value(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, l1, l2, 0, 0, 0))
        cands.append((_law_label(lid, cost, [l1, l2]), v))
        acts.append(("law", li, l1, l2))
        if l1 >= 0 or l2 >= 0:
            v = float(action_value(si, sf, theta, cfg, K, R, sb, KIND_LAW, li, -1, -1, 0, 0, 0))
            cands.append((_law_label(lid, cost, [-1, -1]) + "（不提前放牌）", v))
            acts.append(("law", li, -1, -1))
    ci = int(np.argmax([v for _, v in cands])) if cands else -1
    return f"第 {t} 回合：滚动前瞻（每个候选各模拟 {R} 局）", "期望总倍率", cands, acts, plan0, ci


def play_expert(game, theta, K=12, R=200):
    """滚动前瞻专家：与 tulu.rl.expert.decide 同一逻辑，记录每个候选动作的模拟平均总倍率。"""
    dec = 0
    gseed = game.seed
    while not game.done() and dec < 400:
        dec += 1
        sb = gseed * 7919 + dec * 104729
        title, unit, cands, acts, plan0, ci = expert_eval(game, theta, K, R, sb)
        if not cands:
            break
        game.decision(title, unit, cands, ci)
        kind, a, l1, l2 = acts[ci]
        if kind == "opt":
            game.pick(a)
        elif kind == "finish":
            game.finish_plan(plan0)
        else:
            if l1 >= 0:
                game.place_form(l1)
            if l2 >= 0:
                game.place_form(l2)
            game.law_idx(a)
        seed_rng(gseed * 1000003 + dec * 7919 + 17)
    return game


def play_ppo(game, net, device="cpu", topk=5):
    """PPO 网络（贪心）：每个原子动作一次决策，候选为概率最高的 topk 个动作。"""
    import torch
    from tulu.rl.vec_env import build_obs_mask, OBS_DIM, NUM_ACTIONS, A_LAW0, A_FREE0, A_OPT0
    si, sf, cfg = game.si, game.sf, game.cfg
    names = action_names()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    guard = 0
    while not game.done() and guard < 400:
        guard += 1
        build_obs_mask(si, sf, cfg, obs, mask)
        if not mask.any():
            break
        with torch.no_grad():
            logits, _ = net.forward(torch.from_numpy(obs)[None].to(device), torch.from_numpy(mask)[None].to(device))
            p = torch.softmax(logits[0], dim=-1).cpu().numpy()
        order = np.argsort(-p)[:topk]
        order = [int(a) for a in order if mask[a] and p[a] > 1e-4] or [int(np.argmax(p))]
        a = order[0]
        cands = [(names[x].replace("（付费）", ""), float(p[x]) * 100) for x in order]
        game.decision(f"第 {int(si[O_T])} 回合：网络给出的动作概率", "概率 %", cands, 0)
        if a < A_LAW0:
            game.place_form(a)
        elif a < A_OPT0:
            k = a - A_LAW0
            game.law_by_id(k % 7, paid=k < 7)
        else:
            lid = a - A_OPT0
            j = next(j for j in range(int(si[O_PN])) if si[O_PO + j] == lid)
            game.pick(j)
    return game
