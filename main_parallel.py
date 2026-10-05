import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool

cur_dir = os.getcwd()
print(cur_dir)
os.chdir('src')

from agent.base_agent import BaseAgent
from algorithm.qpsp import QPSPAlgorithm

from logfunc.callback import LoggerCallback

ENV_NUM = 4
AGENT_NUM = 1
ALPHA = 0.06
GAMMA = 0.95
D_RATE = 0.20
TEMPERATURE = 0.52
PAST_N = 4
#PAST_N = None
MAX_STEPS = 500000
BLOCK = 1000
SEEDS = [0]
#SEEDS = [0,1,2,3,4,5,6,7,8,9]

IsMomentum = False
Lambda = 0.9

if IsMomentum is True:
    from gridworld.environment_momentum import Environment
else:
    from gridworld.environment import Environment

momentum_str = "acthistory" if IsMomentum else "relative"

#コアごとに並列実行される関数
def run_simulation(args):
    seed, outf = args
    print(f"seed={seed} start", flush=True)
    random.seed(seed)
    np.random.seed(seed)

    log_file_path = os.path.join(outf, f'env={ENV_NUM}_agent={AGENT_NUM}_N={PAST_N}_seed={seed}.txt')

    if IsMomentum is True:
        env = Environment(AGENT_NUM, ENV_NUM, PAST_N, lam=Lambda)
    else:
        env = Environment(AGENT_NUM, ENV_NUM, PAST_N)
        
    agents = [BaseAgent() for _ in range(AGENT_NUM)]
    algo = QPSPAlgorithm(alpha=ALPHA, gamma=GAMMA, d_rate=D_RATE)

    callback = LoggerCallback(log_file_path, AGENT_NUM, PAST_N)

    history = algo.run(env, agents, TEMPERATURE, MAX_STEPS, callback=callback, block=BLOCK)

    callback.on_seed_end(seed, history)

    print(f"seed{seed} done", flush=True)
    return seed, history

if __name__ == '__main__':
    OUTF = './result'
    now = datetime.now()
    NOW = now.strftime("%Y%m%d-%H%M")
    OUTF = os.path.join(OUTF,NOW)

    if not os.path.exists(OUTF):
        os.makedirs(OUTF)

    print(f" 設定: 環境：{ENV_NUM} / エージェント:{AGENT_NUM}体 / 相対ベクトル: {PAST_N} / ベクトルモード：{momentum_str} / ステップ数：{MAX_STEPS}")

    task_args = [(seed, OUTF) for seed in SEEDS]
    
    with Pool(processes=5) as pool:
        results = pool.map(run_simulation, task_args)

    print("\n計算完了。グラフ一括プロットします。")
    
    plt.figure()
    results.sort(key=lambda x: x[0])

    for seed, history in results:
        steps = [(i+1) * BLOCK for i in range(len(history))]
        plt.plot(steps, history, label=f"seed={seed}")
    plt.xlabel("step")
    plt.ylabel("deliveries per 1000 steps")
    plt.ylim(top=300)
    
    plt.title(f"agents={AGENT_NUM}")

    plt.grid(True,linestyle="--", alpha=0.6)

    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0), borderaxespad=0)
    plt.tight_layout()

    graph_path = os.path.join(OUTF, f'env_{ENV_NUM}_agents={AGENT_NUM}_n={PAST_N}.png')
    plt.savefig(graph_path)
    print(f"保存完了:{graph_path}")

    print("テキストファイルを一つにまとめます")
    combined_log_path = os.path.join(OUTF, f'env={ENV_NUM}_agent={AGENT_NUM}_N={PAST_N}_{momentum_str}.txt')
    
    with open(combined_log_path, 'a', encoding='utf-8') as combined_f:
        for seed in SEEDS:
            individual_file_path = os.path.join(OUTF, f'env={ENV_NUM}_agent={AGENT_NUM}_N={PAST_N}_seed={seed}.txt')

            if os.path.exists(individual_file_path):
                with open(individual_file_path, 'r', encoding='utf-8') as f:
                    combined_f.write(f.read())

                os.remove(individual_file_path)
    print("テキストファイルまとめました")
                    
