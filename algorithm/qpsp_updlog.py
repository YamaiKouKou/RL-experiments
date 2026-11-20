"""QPSPLogAlgorithm に「Q 更新の中身」の記録を足した版。学習の中身は qpsp.py と同じ。

搬出して Q 値を更新するたびに、そのエピソードで使った (状態, 行動) 1 つ 1 つについて
「どれだけ報酬が割り振られ、Q 値がどれだけ変わったか」を 1 行ずつ記録する。

1 行 = 1 回の Q 更新（エージェントごと）
  deliv_step : 搬出したステップ（この時に更新が走る）
  act_step   : その行動をとったステップ（= deliv_step - i）
  agent      : エージェント番号
  i          : 搬出の何手前の行動か（0 = 搬出した行動そのもの）
  ep_len     : そのエピソードの長さ（更新した行の数）
  obs        : 観測 8 マス（北西, 北, 北東, 西, 東, 南西, 南, 南東）
  cargo      : 荷物の有無
  vy, vx     : 状態に入っている v（None なら NaN）
  action     : 記録された行動 0上 1下 2左 3右 4停止
  r_alloc    : 割り振られた報酬 r·d^i（i = 0 は r）
  next_term  : γ·maxQ(次の状態)（i = 0 は γ·r/(1-γ)）
  target     : 目標値 = r_alloc + next_term
  q_before   : 更新前の Q 値
  q_after    : 更新後の Q 値（ΔQ = q_after - q_before = α(target - q_before)）
更新は i = 0（搬出した行動）から過去へ向かって順に計算される。
同じエピソードで同じ (状態, 行動) を何度もとっていれば、その回数だけ行がある。

使い方: QPSPLogAlgorithm の代わりにこのクラスを使い、upd_until を渡す。
  algo = QPSPUpdLogAlgorithm(..., upd_until=1_000_000)   # 搬出ステップが 100万未満の更新を記録
保存: algo.save(".../xxx_log.npz") で、いつもの _log.npz に加えて _updates.npz も書き出す。
"""
import numpy as np
from algorithm.qpsp_log import QPSPLogAlgorithm


class _StepCounter:
    """run の中の「今のステップ」を知るために callback を包む（qpsp_log.py は書き換えない）。
    on_step_end(step) が呼ばれた直後から次のステップなので、更新中の self._step は今のステップになる。"""

    def __init__(self, inner, owner):
        self._inner = inner
        self._owner = owner

    def on_step_end(self, step, env, agents):
        self._owner._step = step + 1
        if self._inner is not None:
            self._inner.on_step_end(step, env, agents)

    def __getattr__(self, name):           # それ以外の呼び出しはそのまま元の callback へ
        if self._inner is None:
            return lambda *a, **k: None
        return getattr(self._inner, name)


class QPSPUpdLogAlgorithm(QPSPLogAlgorithm):
    def __init__(self, *args, upd_until=1_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.upd_until = upd_until

    def run(self, env, agents, temperature, max_steps, callback=None, block=1000):
        self._step = 0
        self._upd = []                      # 搬出 1 回ぶんの配列をためていき、保存時につなげる
        return super().run(env, agents, temperature, max_steps,
                           callback=_StepCounter(callback, self), block=block)

    def update(self, agent, reward):
        """qpsp.py の update と同じ計算。各行の中身を記録するだけ追加。"""
        aid = self._agent_id
        step = self._step
        record = step < self.upd_until
        ideal_q = reward / (1 - self.gamma)
        rev = list(reversed(agent.episode_buffer))
        n = len(rev)
        if record:
            r_alloc = np.empty(n, np.float32); nxt = np.empty(n, np.float32)
            qb = np.empty(n, np.float32); qa = np.empty(n, np.float32)
        for i, (state_key, action) in enumerate(rev):
            if i == 0:
                r_t = reward
                next_term = self.gamma * ideal_q
            else:
                r_t = reward * (self.d_rate ** i)
                next_state_key, _ = rev[i - 1]
                next_term = self.gamma * np.max(agent.q_table[next_state_key])
            value = r_t + next_term
            current_q = agent.q_table[state_key][action]
            delta = self.alpha * (value - current_q)
            agent.q_table[state_key][action] = current_q + delta
            self._dq[aid] += abs(delta)
            self._nu[aid] += 1
            if record:
                r_alloc[i] = r_t; nxt[i] = next_term; qb[i] = current_q; qa[i] = current_q + delta
        if record:
            obs = np.array([k[0] for k, _ in rev], np.int16).reshape(n, -1)
            cargo = np.array([bool(k[1]) for k, _ in rev], np.int8)
            vec = np.array([(np.nan, np.nan) if k[2] is None else (float(k[2][0]), float(k[2][1]))
                            for k, _ in rev], np.float32).reshape(n, 2)
            ii = np.arange(n, dtype=np.int32)
            self._upd.append({
                "deliv_step": np.full(n, step, np.int32), "act_step": (step - ii).astype(np.int32),
                "agent": np.full(n, aid, np.int8), "i": ii.astype(np.int16), "ep_len": np.full(n, n, np.int16),
                "obs": obs, "cargo": cargo, "vy": vec[:, 0], "vx": vec[:, 1],
                "action": np.array([a for _, a in rev], np.int8),
                "r_alloc": r_alloc, "next_term": nxt, "target": r_alloc + nxt, "q_before": qb, "q_after": qa})

    def save(self, path_npz, path_ckpt=None, meta=None):
        super().save(path_npz, path_ckpt=path_ckpt, meta=meta)
        if not self._upd:
            return
        keys = self._upd[0].keys()
        arrays = {f"upd_{k}": np.concatenate([u[k] for u in self._upd]) for k in keys}
        arrays.update({"upd_until": np.asarray(self.upd_until), "upd_alpha": np.asarray(self.alpha),
                       "upd_gamma": np.asarray(self.gamma), "upd_d_rate": np.asarray(self.d_rate)})
        if meta:
            arrays.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        path = str(path_npz)
        path = path[:-len("_log.npz")] + "_updates.npz" if path.endswith("_log.npz") else path + "_updates.npz"
        np.savez_compressed(path, **arrays)
