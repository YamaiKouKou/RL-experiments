import os
import time

class LoggerCallback:
    def __init__(self, file_path, agent_num, past_n):
        self.file_path = file_path
        self.agent_num = agent_num
        self.past_n = past_n

        self.start_time = time.time()

    def on_step_end(self, step, env, agents):
        pass

    def on_block_end(self, step, block_count, history):
        pass

    def on_debug_log(self, step, env, agents):
        elapsed = time.time() - self.start_time

        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"--- step={step:>7} | elapsed={elapsed:7.1f}s ---\n")
            f.write(f"pos={env.agents_pos} | cargo={env.agents_cargo}\n" )

            for agent_id, agent in enumerate(agents):
                cur_relvec = env.get_relvec(agent_id)
                state_key = agent.convert_state(env.get_obs(agent_id), env.agents_cargo[agent_id], cur_relvec)

                if state_key in agent.q_table:
                    q_vals = agent.q_table[state_key]
                    f.write(f"  Agent {agent_id} Q-values: [上={q_vals[0]:>6.3f}, 下={q_vals[1]:>6.3f}, 左={q_vals[2]:>6.3f}, 右={q_vals[3]:>6.3f}, 停={q_vals[4]:>6.3f}]\n")
                else:
                    f.write(f"  Agent {agent_id} Q-values: まだ未踏の状態（Q値なし）\n")
            f.write("----------------------------------------------------------\n")

    def on_seed_end(self, seed, history):
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f'{history}\nseed={seed} done\n')
