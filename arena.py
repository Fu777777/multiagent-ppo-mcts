"""自动化对战测试：随机 / 纯MCTS / PPO / PPO+MCTS 两两对战，输出胜率表（先后手各半）。

python arena.py --games 40 --sims 100
"""
import argparse
import itertools
import random
import time

import numpy as np
import torch

from mappo.game import COLS, Connect4


def play_match(a, b, games, opening=2, seed=0):
    """a 与 b 对战 games 局，返回 a 的 (胜, 平, 负)。
    确定性策略互下只会产生 2 盘不同的棋，所以每对棋局先随机走 opening 步作为开局，
    同一个开局 a、b 各执先手一次，保证公平且结果有统计意义。"""
    rng = random.Random(seed)
    w = d = l = 0
    for i in range(games):
        first, second = (a, b) if i % 2 == 0 else (b, a)
        if i % 2 == 0:
            cols = [rng.randrange(COLS) for _ in range(opening)]
        g = Connect4()
        for c in cols:
            g.play(c)
        while g.winner is None:
            g.play((first if g.player == 1 else second).act(g))
        if g.winner == 0:
            d += 1
        elif (g.winner == 1) == (first is a):
            w += 1
        else:
            l += 1
    return w, d, l


def main():
    from mappo.agents import MCTSPlayer, PPOPlayer, RandomPlayer
    from mappo.net import PolicyValueNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/ppo.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--opening", type=int, default=2, help="每局开头随机走几步")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(1)
    net = PolicyValueNet(); net.load_state_dict(torch.load(args.model)); net.eval()
    players = [RandomPlayer(), MCTSPlayer(None, args.sims), PPOPlayer(net), MCTSPlayer(net, args.sims)]
    names = [p.name for p in players]
    table = {(x, x): "-" for x in names}
    t0 = time.time()
    for a, b in itertools.combinations(players, 2):
        w, d, l = play_match(a, b, args.games, args.opening, args.seed)
        print(f"{a.name:14s} vs {b.name:14s}  胜/平/负 {w}/{d}/{l}  ({time.time() - t0:.0f}s)", flush=True)
        table[(a.name, b.name)] = f"{(w + 0.5 * d) / args.games:.0%}"
        table[(b.name, a.name)] = f"{(l + 0.5 * d) / args.games:.0%}"
    print(f"\n胜率表（行 vs 列，平局记半胜，每对 {args.games} 局，先后手各半，开局随机 {args.opening} 步）")
    print("| 行\\列 | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for r in names:
        print(f"| {r} | " + " | ".join(table[(r, c)] for c in names) + " |")


if __name__ == "__main__":
    main()
