"""Policy-value network for Ver2.

Scores a *variable* number of legal options. Each option is scored from
[global board features ++ option features], then a legality mask zeroes out
padded slots before the softmax -- so the policy is, by construction, a
distribution over legal actions only. A separate value head predicts the
game outcome from the global state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoding import global_dim, option_dim, MAX_OPTIONS


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        g_in, o_in = global_dim(), option_dim()
        self.g_enc = _mlp([g_in, hidden, hidden])
        self.o_enc = _mlp([o_in, hidden, hidden])
        self.scorer = _mlp([2 * hidden, hidden, 1])
        self.value = _mlp([hidden, hidden, 1])
        self.hidden = hidden

    def forward(self, glob, options, mask):
        """glob:[B,G]  options:[B,M,F]  mask:[B,M] bool -> logits:[B,M], value:[B]."""
        g = self.g_enc(glob)                       # [B,H]
        o = self.o_enc(options)                    # [B,M,H]
        g_exp = g.unsqueeze(1).expand(-1, o.size(1), -1)
        logits = self.scorer(torch.cat([g_exp, o], dim=-1)).squeeze(-1)  # [B,M]
        logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value(g)).squeeze(-1)                    # [B] in (-1,1)
        return logits, value

    @torch.no_grad()
    def policy_value(self, glob, options, mask, device):
        """Single-decision inference. numpy in -> (probs np[n], value float)."""
        n = int(mask.sum())
        gt = torch.as_tensor(glob, dtype=torch.float32, device=device).unsqueeze(0)
        ot = torch.as_tensor(options, dtype=torch.float32, device=device).unsqueeze(0)
        mt = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value = self.forward(gt, ot, mt)
        probs = F.softmax(logits[0, :n], dim=-1).cpu().numpy()
        return probs, float(value.item())


def pad_decision(ed, max_options=MAX_OPTIONS):
    """EncodedDecision -> (glob np[G], options np[M,F], mask np[M] bool)."""
    import numpy as np
    o = ed.options
    M, Fdim = max_options, option_dim()
    padded = np.zeros((M, Fdim), dtype=np.float32)
    mask = np.zeros((M,), dtype=bool)
    n = min(ed.n_options, M)
    if n > 0:
        padded[:n] = o[:n]
        mask[:n] = True
    return ed.glob, padded, mask
