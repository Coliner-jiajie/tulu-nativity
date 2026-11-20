"""多个检查点的集成：对各网络的动作概率取平均后再取最大。可直接传给 evaluate_net。"""
import torch


class EnsembleNet:
    def __init__(self, nets, weights=None):
        self.nets = nets
        n = len(nets)
        self.w = [1.0 / n] * n if weights is None else [x / sum(weights) for x in weights]

    def eval(self):
        for n in self.nets:
            n.eval()
        return self

    @torch.no_grad()
    def act(self, obs, mask, greedy=True):
        probs = None
        for net, w in zip(self.nets, self.w):
            logits, _ = net(obs, mask)
            p = torch.softmax(logits, dim=-1) * w
            probs = p if probs is None else probs + p
        if greedy:
            a = probs.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(probs=probs).sample()
        logp = torch.log(probs.gather(-1, a[:, None]).squeeze(-1) + 1e-12)
        return a, logp, torch.zeros_like(logp)
