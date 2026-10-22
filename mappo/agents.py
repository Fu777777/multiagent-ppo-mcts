"""对战用的四种智能体，统一接口 act(game) -> 列号。"""
import random

import numpy as np

from .mcts import MCTS


class RandomPlayer:
    name = "随机"

    def act(self, game):
        return int(random.choice(game.legal_moves()))


class PPOPlayer:
    """直接取策略网络概率最大的合法列（贪心）。"""
    name = "PPO"

    def __init__(self, net):
        self.net = net

    def act(self, game):
        p, _ = self.net.predict(game)
        return int(np.argmax(p))


class MCTSPlayer:
    def __init__(self, net=None, sims=100):
        self.mcts = MCTS(net, sims)
        self.name = f"PPO+MCTS({sims})" if net is not None else f"纯MCTS({sims})"

    def act(self, game):
        return self.mcts.act(game)
