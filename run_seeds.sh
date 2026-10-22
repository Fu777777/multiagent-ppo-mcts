#!/usr/bin/env bash
# 多种子复核：seed 1、2 各训练 1000 轮，再各跑 arena（100 局、100 次模拟）；seed 0 用已有模型复跑 arena 检验可复现
cd "$(dirname "$0")"
source /workspace/doudizhu-rl/.venv/bin/activate
mkdir -p logs_seeds
run() { local log=$1; shift; local t0=$SECONDS; python "$@" > logs_seeds/$log 2>&1; echo "WALL $((SECONDS-t0)) s | cmd: python $*" >> logs_seeds/$log; }
run arena_s0_rerun.txt arena.py --model checkpoints/ppo.pt --games 100 --sims 100 &
for s in 1 2; do
  ( run train_s$s.txt train.py --iters 1000 --games 64 --seed $s --out checkpoints/ppo_s$s.pt && \
    run arena_s$s.txt arena.py --model checkpoints/ppo_s$s.pt --games 100 --sims 100 ) &
done
wait; echo ALL_DONE
