"""PPO 自对弈训练入口。python train.py --iters 150 --games 64"""
import argparse
import os
import random
import time

import numpy as np
import torch

from mappo.agents import PPOPlayer, RandomPlayer
from mappo.net import PolicyValueNet
from mappo.ppo import ppo_update, self_play
from arena import play_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--games", type=int, default=64, help="每轮自对弈局数")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/ppo.pt", help="模型保存路径（多种子实验用）")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    net = PolicyValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        batch = self_play(net, args.games, rng=rng)
        pl, vl, ent = ppo_update(net, opt, batch)
        if it % 25 == 0:
            w, d, l = play_match(PPOPlayer(net), RandomPlayer(), 100)
            print(f"iter {it:4d} | 样本 {len(batch[0]):5d} | 策略损失 {pl:+.3f} | 价值损失 {vl:.3f} | 熵 {ent:.3f} | "
                  f"贪心PPO vs 随机(100局) 胜/平/负 {w}/{d}/{l} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print("模型已保存:", args.out)


if __name__ == "__main__":
    main()
