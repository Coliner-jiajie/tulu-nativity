"""方法 B：演员-评论家网络（需要 PyTorch）。策略头输出 36 维动作 logits，非法动作用掩码置为 -1e9。"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tulu.rl.vec_env import NUM_ACTIONS, OBS_DIM

NEG_INF = -1e9


def _init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=256, layers=2):
        super().__init__()
        mods = []
        d = obs_dim
        for _ in range(layers):
            mods += [_init(nn.Linear(d, hidden), 2 ** 0.5), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*mods)
        self.pi = _init(nn.Linear(hidden, n_actions), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.meta = {"obs_dim": obs_dim, "n_actions": n_actions, "hidden": hidden, "layers": layers}

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, NEG_INF)
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, mask, greedy=False):
        logits, value = self.forward(obs, mask)
        if greedy:
            a = logits.argmax(dim=-1)
            logp = torch.log_softmax(logits, dim=-1).gather(-1, a[:, None]).squeeze(-1)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        return a, logp, value

    def evaluate_actions(self, obs, mask, actions):
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DuelingQNet(nn.Module):
    """决斗式 Q 网络：Q(s,a) = V(s) + A(s,a) − mean_{合法 a} A(s,a)。非法动作的 Q 置为 -1e9。
    贪心动作 = 合法动作中 Q 最大者；接口与 ActorCritic 兼容（act / forward），可直接交给 evaluate_net。
    Q 的单位：剩余总倍率 / 100。"""

    def __init__(self, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=512, layers=3, type="dueling"):
        super().__init__()
        mods = []
        d = obs_dim
        for _ in range(layers):
            mods += [_init(nn.Linear(d, hidden), 2 ** 0.5), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*mods)
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.a = _init(nn.Linear(hidden, n_actions), 0.1)
        self.meta = {"obs_dim": obs_dim, "n_actions": n_actions, "hidden": hidden, "layers": layers,
                     "type": "dueling"}

    def q_values(self, obs, mask):
        h = self.trunk(obs)
        v = self.v(h)
        a = self.a(h)
        m = mask.float()
        a_mean = (a * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1.0)
        q = v + a - a_mean
        return q.masked_fill(~mask, NEG_INF), v.squeeze(-1)

    def forward(self, obs, mask):
        return self.q_values(obs, mask)

    @torch.no_grad()
    def act(self, obs, mask, greedy=True):
        q, v = self.q_values(obs, mask)
        a = q.argmax(dim=-1)
        return a, torch.zeros_like(v), v
