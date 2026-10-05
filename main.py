import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, 'src'))

from agent.base_agent import BaseAgent
from algorithm.qpsp import QPSPAlgorithm

from logfunc.callback import LoggerCallback

ENV_NUM = 4
AGENT_NUM = 1
ALPHA = 0.06
GAMMA = 0.95
D_RATE = 0.20
TEMPERATURE = 0.52
#PAST_N = 4
PAST_N = None
MAX_STEPS = 5000000
BLOCK = 1000
SEEDS = [0]
#SEEDS = [0,1,2,3,4,5,6,7,8,9]

OUTF = './result'
now = datetime.now()
NOW = now.strftime("%Y%m%d-%H%M")
OUTF = os.path.join(OUTF,NOW)

IsMomentum = True
Lambda = 0.9

if IsMomentum is True:
    from gridworld.environment_momentum import Environment
else:
    from gridworld.environment import Environment

try:
    os.makedirs(OUTF)
except FileExistsError:
    pass

momentum_str = "momentum" if IsMomentum else "normal"
log_file_path = os.path.join(OUTF,f'result_env={ENV_NUM}_agent={AGENT_NUM}_pastN={PAST_N}_{momentum_str}.txt')

plt.figure()

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    if IsMomentum is True:
        env = Environment(AGENT_NUM, ENV_NUM, PAST_N, lam=Lambda)
    else:
        env = Environment(AGENT_NUM, ENV_NUM, PAST_N)
        
    agents = [BaseAgent() for _ in range(AGENT_NUM)]
    algo = QPSPAlgorithm(alpha=ALPHA, gamma=GAMMA, d_rate=D_RATE)

    callback = LoggerCallback(log_file_path, AGENT_NUM, PAST_N)

    history = algo.run(env, agents, TEMPERATURE, MAX_STEPS, callback, block=BLOCK)

    callback.on_seed_end(seed, history)

    q_tables = [agent.q_table for agent in agents]
    with open(f'{OUTF}/qtables_env{ENV_NUM}_agent{AGENT_NUM}_n{PAST_N}_{momentum_str}_seed{seed}.pkl', 'wb') as f:
        pickle.dump(q_tables, f)

    steps = [(i+1) * BLOCK for i in range(len(history))]

    label = f"seed={seed}"
    plt.plot(steps, history, label=label)

plt.xlabel("step")
plt.ylabel("deliveries per 1000 steps")
plt.ylim(top=300)

plt.title(f"agents={AGENT_NUM}")
plt.grid(True,linestyle="--", alpha=0.6)
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0), borderaxespad=0)
plt.tight_layout()

plt.savefig(f'{OUTF}/result_env_{ENV_NUM}_agents={AGENT_NUM}_n={PAST_N}_{momentum_str}.png')
print("シミュレーション全終了")
plt.show()
