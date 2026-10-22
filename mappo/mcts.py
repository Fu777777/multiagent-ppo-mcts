"""MCTS（PUCT 选择）：
- AlphaZero 式：先验 P 与叶子价值 V 都来自策略-价值网络；
- 纯 MCTS 基线：均匀先验 + 随机 rollout 估值（不依赖网络）。
价值一律存为“该节点上刚落子那一方”的视角，回传时逐层取反（负极大值）。
"""
import math
import random

import numpy as np

from .game import COLS


class Node:
    __slots__ = ("prior", "children", "N", "W")

    def __init__(self, prior):
        self.prior, self.children, self.N, self.W = prior, {}, 0, 0.0

    def Q(self):
        return self.W / self.N if self.N else 0.0


def rollout_value(game):
    """随机下到终局，返回对“当前要走的一方”的结果。"""
    g, me = game.copy(), game.player
    while g.winner is None:
        g.play(random.choice(g.legal_moves()))
    return 0.0 if g.winner == 0 else (1.0 if g.winner == me else -1.0)


class MCTS:
    def __init__(self, net=None, sims=100, c_puct=1.5):
        self.net, self.sims, self.c = net, sims, c_puct

    def evaluate(self, game):
        """返回 (先验, 当前方视角价值)。"""
        if self.net is not None:
            return self.net.predict(game)
        m = game.legal_mask()
        return m / m.sum(), rollout_value(game)

    def search(self, game):
        """返回根节点各列访问次数分布。"""
        root = Node(1.0)
        self._expand(root, game)
        for _ in range(self.sims):
            node, g, path = root, game.copy(), [root]
            while node.children and g.winner is None:   # 1) 选择：PUCT
                sqrt_n = math.sqrt(node.N)
                a, node = max(node.children.items(),
                              key=lambda kv: kv[1].Q() + self.c * kv[1].prior * sqrt_n / (1 + kv[1].N))
                g.play(a); path.append(node)
            if g.winner is not None:                    # 2) 终局：刚落子一方赢则对当前方为 -1
                v = 0.0 if g.winner == 0 else -1.0
            else:                                       # 3) 扩展 + 网络/rollout 估值
                v = self._expand(node, g)
            for n in reversed(path):                    # 4) 回传：节点价值属于“走到该节点的一方”
                n.N += 1; n.W += -v; v = -v
        visits = np.zeros(COLS)
        for a, ch in root.children.items():
            visits[a] = ch.N
        return visits

    def _expand(self, node, game):
        p, v = self.evaluate(game)
        for a in game.legal_moves():
            node.children[a] = Node(p[a])
        return v

    def act(self, game):
        return int(np.argmax(self.search(game)))
