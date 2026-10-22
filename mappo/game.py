"""简化版 Connect-4（四子棋）：5 行 x 6 列，连成 4 子获胜，棋子从上往下落。

两个玩家 +1 / -1 轮流落子。网络总是从“当前要走的一方”的视角看棋盘，
因此同一个网络可以同时扮演双方（参数共享的多智能体自对弈）。
"""
import numpy as np

ROWS, COLS, WIN = 5, 6, 4
OBS_DIM = 2 * ROWS * COLS


class Connect4:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1          # 当前要落子的一方
        self.winner = None       # None=未结束, 0=平局, ±1=胜者
        self.moves = 0

    def copy(self):
        g = Connect4.__new__(Connect4)
        g.board, g.player, g.winner, g.moves = self.board.copy(), self.player, self.winner, self.moves
        return g

    def legal_mask(self):
        return self.board[0] == 0  # 顶行为空的列可落子

    def legal_moves(self):
        return np.flatnonzero(self.legal_mask())

    def play(self, col):
        assert self.winner is None and self.board[0, col] == 0, "非法落子"
        row = ROWS - 1 - np.argmax(self.board[::-1, col] == 0)  # 最低的空位
        self.board[row, col] = self.player
        self.moves += 1
        if self._wins(row, col):
            self.winner = self.player
        elif self.moves == ROWS * COLS:
            self.winner = 0
        self.player = -self.player

    def _wins(self, r, c):
        """只检查刚落的子所在的四个方向。"""
        p = self.board[r, c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                rr, cc = r + s * dr, c + s * dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == p:
                    n += 1; rr += s * dr; cc += s * dc
            if n >= WIN:
                return True
        return False

    def obs(self):
        """当前方视角编码：第 1 平面为自己的子，第 2 平面为对方的子。"""
        b = self.board * self.player
        return np.concatenate([(b == 1).ravel(), (b == -1).ravel()]).astype(np.float32)

    def __str__(self):
        sym = {1: "X", -1: "O", 0: "."}
        return "\n".join(" ".join(sym[v] for v in row) for row in self.board) + "\n" + " ".join(map(str, range(COLS)))
