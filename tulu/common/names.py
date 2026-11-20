"""星象、法则、动作的中文名称与解析。"""
STAR_TYPE = ["灾", "垣", "蕴"]
LAW_NAMES = ["易象", "增辉", "敛芒", "昭示", "引辰", "拓途", "重衍"]
STAR_NAMES = [f"{STAR_TYPE[k]}{r}" for k in range(3) for r in range(1, 6)]   # 编号 id = k*5 + (r-1)


def parse_card(s):
    """'灾3' -> 2；'蕴5' -> 14。"""
    k = STAR_TYPE.index(s[0])
    r = int(s[1:])
    assert 1 <= r <= 5
    return k * 5 + (r - 1)


def parse_law(s):
    return LAW_NAMES.index(s)


def action_names():
    """36 个动作：0~14 放入某种星象；15~21 打出消耗 5 的法则；22~28 打出消耗 0 的法则；29~35 昭示发现项选择。"""
    names = [f"放入 {n}" for n in STAR_NAMES]
    names += [f"打出 {n}（付费）" for n in LAW_NAMES]
    names += [f"打出 {n}（0 消耗）" for n in LAW_NAMES]
    names += [f"昭示选取 {n}" for n in LAW_NAMES]
    return names
