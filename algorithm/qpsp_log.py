"""QPSPAlgorithm に解析用の記録を足した版。学習の中身は qpsp.py と同じ。

記録するもの
  ブロック(1000ステップ)ごと × エージェントごと
    dq_mean   : そのブロックで行った Q 更新 1 回あたりの |ΔQ| の平均（更新なしは NaN）
    dq_sum    : そのブロックの |ΔQ| の合計
    n_update  : Q 更新の回数（= 学習に使った (s,a) の数）
    n_deliv   : 搬出回数
    n_discard : 2000 ステップで記録を捨てた回数
    n_states  : ブロック終了時点の Q テーブルの状態数
  エピソードごと（save_episodes=True のとき）
    step, agent, length, delivered(1)/discarded(0),
    q_start : エピソード最初の状態の max_a Q（更新前）
    q_max   : エピソード中に訪れた状態の max_a Q の最大値（更新前）
  チェックポイントごと（ckpt_every ステップ）: 各エージェントの Q テーブルのコピー
  軌跡（traj_windows を指定したときだけ）: 指定区間の全ステップ × 全エージェント
    pos    : 行動する直前の座標 (y, x)
    cargo  : 行動する直前の荷物の有無
    vec    : 行動する直前の相対ベクトル / 行動履歴ベクトル（使わない手法は (0,0)）
    q      : 行動する直前の状態の Q 値（上,下,左,右,停止）
    action : 実際の行動 0上 1下 2左 3右 4停止
    deliv  : その行動で搬出したか（1 のとき、次の pos は出現位置に飛ぶ）
"""
import copy
import numpy as np
from algorithm.qpsp import QPSPAlgorithm


def _open_w(path):
    """末尾が .gz なら gzip で圧縮して書く（中身は同じ pickle。gzip.open で読めば元どおり）。"""
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "wb")
    return open(path, "wb")


class QPSPLogAlgorithm(QPSPAlgorithm):
    def __init__(self, alpha=0.1, gamma=0.9, d_rate=0.2,
                 save_episodes=True, ckpt_every=None, traj_windows=None, ckpt_path=None):
        super().__init__(alpha=alpha, gamma=gamma, d_rate=d_rate)
        self.save_episodes = save_episodes
        self.ckpt_every = ckpt_every
        # ckpt_path を渡すと、チェックポイントのたびにその場で pickle に書き出す（メモリに溜めない）
        # 例: "out/tag_seed=0_qtable_step={step}.pkl.gz"（末尾が .gz なら gzip で圧縮して保存）
        self.ckpt_path = ckpt_path
        # 例: [(0, 20000), (2_490_000, 2_510_000), (4_980_000, 5_000_000)]  終端は含まない
        self.traj_windows = sorted(traj_windows) if traj_windows else []

    def run(self, env, agents, temperature, max_steps, callback=None, block=1000):
        env.reset()
        for agent in agents:
            agent.reset_history()

        A = len(agents)
        nb = max_steps // block
        self.log = {k: np.zeros((nb, A), dtype=dt) for k, dt in [
            ("dq_sum", np.float64), ("n_update", np.int32), ("n_deliv", np.int32),
            ("n_discard", np.int32), ("n_states", np.int32)]}
        ep = {k: [] for k in ("step", "agent", "length", "delivered", "q_start", "q_max")}
        self.ckpts = {}

        # エピソード中の作業用
        q_start = [np.nan] * A
        q_max = [-np.inf] * A
        self._dq = np.zeros(A)
        self._nu = np.zeros(A, dtype=np.int64)
        n_deliv = np.zeros(A, dtype=np.int64)
        n_disc = np.zeros(A, dtype=np.int64)

        # 軌跡の記録先
        self.traj = []
        for (s0, s1) in self.traj_windows:
            s1 = min(s1, max_steps); L = max(s1 - s0, 0)
            self.traj.append({"start": s0, "end": s1,
                              "pos": np.zeros((L, A, 2), np.int8),
                              "cargo": np.zeros((L, A), np.int8),
                              "vec": np.zeros((L, A, 2), np.float32),
                              "q": np.zeros((L, A, 5), np.float32),
                              "action": np.zeros((L, A), np.int8),
                              "deliv": np.zeros((L, A), np.int8)})
        wi = 0   # 次に来る(または今の)区間の番号

        history = []
        block_count = 0

        for step in range(max_steps):
            while wi < len(self.traj) and step >= self.traj[wi]["end"]:
                wi += 1
            tr = self.traj[wi] if (wi < len(self.traj) and step >= self.traj[wi]["start"]) else None
            k = step - tr["start"] if tr is not None else None

            for agent_id, agent in enumerate(agents):
                agent.obs = env.get_obs(agent_id)
                agent.cargo = env.agents_cargo[agent_id]
                rel_vec = env.get_relvec(agent_id)
                state_key = agent.convert_state(agent.obs, agent.cargo, rel_vec)
                action = agent.get_action(temperature, rel_vec)   # ここで状態が初期化される

                # 更新前の Q 値を記録
                qmax_s = float(np.max(agent.q_table[state_key]))
                if not agent.episode_buffer:
                    q_start[agent_id] = qmax_s
                    q_max[agent_id] = qmax_s
                elif qmax_s > q_max[agent_id]:
                    q_max[agent_id] = qmax_s

                if tr is not None:
                    tr["pos"][k, agent_id] = env.agents_pos[agent_id]
                    tr["cargo"][k, agent_id] = bool(agent.cargo)
                    tr["vec"][k, agent_id] = rel_vec if rel_vec is not None else (0, 0)
                    tr["q"][k, agent_id] = agent.q_table[state_key]

                reward, delivered, movable = env.step(agent_id, action)

                if tr is not None:
                    tr["action"][k, agent_id] = action if movable else 4
                    tr["deliv"][k, agent_id] = bool(delivered)
                agent.episode_buffer.append((state_key, action if movable else 4))

                ended = None
                if delivered:
                    self._agent_id = agent_id
                    self.update(agent, reward)
                    ended = 1
                    n_deliv[agent_id] += 1
                    block_count += 1
                elif len(agent.episode_buffer) >= 2000:
                    ended = 0
                    n_disc[agent_id] += 1

                if ended is not None:
                    if self.save_episodes:
                        ep["step"].append(step)
                        ep["agent"].append(agent_id)
                        ep["length"].append(len(agent.episode_buffer))
                        ep["delivered"].append(ended)
                        ep["q_start"].append(q_start[agent_id])
                        ep["q_max"].append(q_max[agent_id])
                    agent.episode_buffer = []

            if callback:
                callback.on_step_end(step, env, agents)

            if (step + 1) % block == 0:
                b = (step + 1) // block - 1
                self.log["dq_sum"][b] = self._dq
                self.log["n_update"][b] = self._nu
                self.log["n_deliv"][b] = n_deliv
                self.log["n_discard"][b] = n_disc
                self.log["n_states"][b] = [len(a.q_table) for a in agents]
                self._dq[:] = 0; self._nu[:] = 0; n_deliv[:] = 0; n_disc[:] = 0

                history.append(block_count)
                if callback:
                    callback.on_block_end(step, block_count, history)
                block_count = 0

            if self.ckpt_every and (step + 1) % self.ckpt_every == 0:
                self._checkpoint(step + 1, agents)

            if callback and step % 100000 == 0:
                callback.on_debug_log(step, env, agents)

        with np.errstate(invalid="ignore", divide="ignore"):
            self.log["dq_mean"] = self.log["dq_sum"] / self.log["n_update"]
        self.episodes = {
            "step": np.asarray(ep["step"], dtype=np.int32),
            "agent": np.asarray(ep["agent"], dtype=np.int8),
            "length": np.asarray(ep["length"], dtype=np.int16),
            "delivered": np.asarray(ep["delivered"], dtype=np.int8),
            "q_start": np.asarray(ep["q_start"], dtype=np.float32),
            "q_max": np.asarray(ep["q_max"], dtype=np.float32),
        }
        return history

    def _checkpoint(self, step, agents):
        if self.ckpt_path:
            import pickle
            with _open_w(self.ckpt_path.format(step=step)) as f:
                pickle.dump([a.q_table for a in agents], f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.ckpts[step] = [copy.deepcopy(a.q_table) for a in agents]

    def update(self, agent, reward):
        """qpsp.py の update と同じ計算。|ΔQ| を足し込むだけ追加。"""
        aid = self._agent_id
        ideal_q = reward / (1 - self.gamma)
        rev = list(reversed(agent.episode_buffer))
        for i, (state_key, action) in enumerate(rev):
            if i == 0:
                value = reward + self.gamma * ideal_q
            else:
                r_t = reward * (self.d_rate ** i)
                next_state_key, _ = rev[i - 1]
                max_q = np.max(agent.q_table[next_state_key])
                value = r_t + (self.gamma * max_q)
            current_q = agent.q_table[state_key][action]
            delta = self.alpha * (value - current_q)
            agent.q_table[state_key][action] = current_q + delta
            self._dq[aid] += abs(delta)
            self._nu[aid] += 1

    def save(self, path_npz, path_ckpt=None, meta=None):
        """ブロック記録とエピソード記録を npz に、チェックポイントを pickle に保存。"""
        arrays = {f"block_{k}": v for k, v in self.log.items()}
        arrays.update({f"ep_{k}": v for k, v in self.episodes.items()})
        for i, tr in enumerate(self.traj):
            for key, v in tr.items():
                arrays[f"traj{i}_{key}"] = np.asarray(v)
        if meta:
            arrays.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        np.savez_compressed(path_npz, **arrays)
        if path_ckpt and self.ckpts:
            import pickle
            with _open_w(path_ckpt) as f:
                pickle.dump(self.ckpts, f, protocol=pickle.HIGHEST_PROTOCOL)
