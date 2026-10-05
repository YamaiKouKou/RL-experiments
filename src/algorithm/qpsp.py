from algorithm.base_algorithm import BaseAlgorithm
import numpy as np

class QPSPAlgorithm(BaseAlgorithm):
    def __init__(self,alpha=0.1,gamma=0.9,d_rate=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.d_rate = d_rate

    def run(self, env, agents, temperature, max_steps, callback=None, block=1000):
        env.reset()
        for agent in agents:
            agent.reset_history()
            
        history = []
        block_count = 0
        
        for step in range(max_steps):
            for agent_id, agent in enumerate(agents):
                
                agent.obs = env.get_obs(agent_id)
                agent.cargo = env.agents_cargo[agent_id]
                
                rel_vec = env.get_relvec(agent_id)
                
                state_key = agent.convert_state(agent.obs, agent.cargo, rel_vec)
                action = agent.get_action(temperature, rel_vec)

                reward,delivered,movable = env.step(agent_id,action)

                if movable:
                    agent.episode_buffer.append((state_key,action))
                else:
                    agent.episode_buffer.append((state_key,4))
                    
                if delivered:
                    self.update(agent,reward)
                    agent.episode_buffer = []
                    block_count += 1
                else:
                    if(len(agent.episode_buffer) >= 2000):
                        agent.episode_buffer = []

            #毎ステップ終了時
            if callback:
                callback.on_step_end(step, env, agents)
            #1000ステップ毎終了時
            if(step+1) % block == 0:
                history.append(block_count)
                callback.on_block_end(step, block_count, history)
                block_count = 0
                
            #10万ごとのデバック
            if step % 100000 == 0:
                callback.on_debug_log(step, env, agents)
                
        return history
                    
    def update(self,agent,reward):
        ideal_q = reward / (1 - self.gamma)

        rev = list(reversed(agent.episode_buffer))

        for i,(state_key,action) in enumerate(rev):
            if i == 0:
                value = reward + self.gamma * ideal_q
            else:
                r_t = reward * (self.d_rate ** i)

                next_state_key,_ = rev[i-1]
                max_q = np.max(agent.q_table[next_state_key])

                value = r_t + (self.gamma * max_q)

            current_q = agent.q_table[state_key][action]
            agent.q_table[state_key][action] = current_q + self.alpha * (value - current_q)
        
        
