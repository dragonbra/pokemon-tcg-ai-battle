"""Recurrent actor-critic network for variable legal options."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoding import global_dim, option_dim


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class A3CLSTMNet(nn.Module):
    def __init__(self, hidden=192, residual_context=False):
        super().__init__()
        self.g_enc = _mlp([global_dim(), hidden, hidden])
        self.o_enc = _mlp([option_dim(), hidden, hidden])
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.scorer = _mlp([2 * hidden, hidden, 1])
        self.value = _mlp([hidden, hidden, 1])
        self.hidden = hidden
        self.residual_context = bool(residual_context)

    def initial_state(self, batch_size=1, device="cpu"):
        h = torch.zeros(batch_size, self.hidden, device=device)
        c = torch.zeros(batch_size, self.hidden, device=device)
        return h, c

    def forward_step(self, glob, options, mask, state):
        """One recurrent decision.

        glob: [B,G], options:[B,M,F], mask:[B,M] bool.
        state: (h,c), each [B,H].
        """
        g = self.g_enc(glob)
        h, c = self.lstm(g, state)
        ctx = g + h if self.residual_context else h
        o = self.o_enc(options)
        ctx_exp = ctx.unsqueeze(1).expand(-1, o.size(1), -1)
        logits = self.scorer(torch.cat([ctx_exp, o], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value(ctx)).squeeze(-1)
        return logits, value, (h, c)

    @torch.no_grad()
    def policy_value_step(self, glob, options, mask, state, device):
        gt = torch.as_tensor(glob, dtype=torch.float32, device=device).unsqueeze(0)
        ot = torch.as_tensor(options, dtype=torch.float32, device=device).unsqueeze(0)
        mt = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value, next_state = self.forward_step(gt, ot, mt, state)
        n = int(mt.sum())
        probs = F.softmax(logits[0, :n], dim=-1).cpu().numpy()
        return probs, float(value.item()), next_state
