"""PPO 自对弈：同一个网络同时控制双方（参数共享），每一步都是从当前方视角的样本。

回报：终局时胜方每一步得 +1、负方每一步得 -1（平局 0），按该玩家自己的步数折扣 gamma。
优势：A = G - V(s)，批内标准化；损失 = 裁剪策略损失 + c_v * 价值 MSE - c_e * 熵。
"""
import numpy as np
import torch
import torch.nn as nn

from .game import Connect4


def self_play(net, n_games, gamma=0.99, rng=None):
    rng = rng or np.random.default_rng()
    S, M, A, LP, G = [], [], [], [], []
    for _ in range(n_games):
        g, traj = Connect4(), {1: [], -1: []}
        while g.winner is None:
            obs, mask = g.obs(), g.legal_mask()
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
                p = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
            a = int(rng.choice(len(p), p=p / p.sum()))
            traj[g.player].append((obs, mask, a, float(np.log(p[a] + 1e-12))))
            g.play(a)
        for pl, steps in traj.items():
            z = 0.0 if g.winner == 0 else (1.0 if g.winner == pl else -1.0)
            T = len(steps)
            for t, (o, m, a, lp) in enumerate(steps):
                S.append(o); M.append(m); A.append(a); LP.append(lp); G.append(z * gamma ** (T - 1 - t))
    return (torch.tensor(np.array(S)), torch.tensor(np.array(M)), torch.tensor(A),
            torch.tensor(LP, dtype=torch.float32), torch.tensor(G, dtype=torch.float32))


def ppo_update(net, opt, batch, epochs=4, mb=256, clip=0.2, c_v=0.5, c_e=0.01):
    S, M, A, LP, G = batch
    with torch.no_grad():
        _, V = net(S, M)
    adv = G - V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n, stats = len(S), []
    for _ in range(epochs):
        for idx in torch.randperm(n).split(mb):
            logits, v = net(S[idx], M[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, A[idx, None]).squeeze(1)
            ratio = torch.exp(logp - LP[idx])
            pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            vl = nn.functional.mse_loss(v, G[idx])
            p = logp_all.exp()
            ent = -(p * logp_all.masked_fill(~M[idx], 0)).sum(-1).mean()
            loss = pl + c_v * vl - c_e * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            stats.append((pl.item(), vl.item(), ent.item()))
    return np.mean(stats, 0)
