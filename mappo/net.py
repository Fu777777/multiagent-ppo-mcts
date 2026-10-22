"""策略-价值网络：共享主干 + 策略头(每列 logits) + 价值头(当前方视角的胜负期望，tanh 到 [-1,1])。"""
import numpy as np
import torch
import torch.nn as nn

from .game import COLS, OBS_DIM


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, COLS)
        self.v = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x, mask):
        h = self.body(x)
        logits = self.pi(h).masked_fill(~mask, -1e9)  # 非法列概率为 0
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, game):
        """给 MCTS / 对战用：返回 (合法动作上的先验概率 np.array[COLS], 价值 float)。"""
        logits, v = self(torch.from_numpy(game.obs())[None], torch.from_numpy(game.legal_mask())[None])
        return torch.softmax(logits, -1)[0].numpy(), float(v[0])
