"""学習済みの Q テーブルで、指定した配置からエージェントを動かす（学習はしない）。

結果は _log.npz と同じ形（traj0_pos など）で保存するので、replay_pygame.py でそのまま見られる。

使い方（main_parallel.py と同じフォルダに置いて実行）
  # 全停止していた配置から、500万ステップ時点の Q テーブルで 2000 ステップ動かす
  python play_qtable.py result/..._seed=0_qtable_step=5000000.pkl.gz \
      --pos "4,4;4,5;4,3" --cargo 0,0,0 --hist "SS;SS;SS" --steps 2000

  # 描画
  python replay_pygame.py result/..._seed=0_qtable_step=5000000_play.npz

引数
  --pos    : 初期位置 "y,x;y,x;..."（エージェント 0, 1, 2 の順）。省略するとランダム
  --cargo  : 荷物の有無 0/1 をカンマ区切り（既定: 全員なし）
  --hist   : スタート前の行動履歴（古い順）。U上 D下 L左 R右 S停止。エージェントごとに ; で区切る
             例 "SS;SS;SS" … 全員 2 回停止した直後（N=2 なら v=(0,0)）
             省略すると、学習のリセット直後と同じ（履歴なし）
  --steps  : 動かすステップ数（既定 2000）
  --seed   : 乱数のシード。省略すると毎回違う動きになる
  --method / --env / --past_n / --lam : 省略するとファイル名から読み取る
  --temp   : Softmax の温度（既定 0.52）
  --out    : 保存先（既定: Q テーブルと同じ場所に _play.npz）

Q テーブルは更新しない（学習しない）。学習中に一度も訪れていない状態に入ると、
その状態の Q 値は全部 0 なので、選べる行動から等確率で選ぶ（学習時と同じ動作）。
そうなったステップは traj0_unseen に 1 が入る。
"""
import argparse
import gzip
import os
import pickle
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # gridworld, agent を読み込めるように
from agent.base_agent import BaseAgent

ACT_CHAR = {"U": 0, "D": 1, "L": 2, "R": 3, "S": 4, "上": 0, "下": 1, "左": 2, "右": 3, "停": 4}
DY = [-1, 1, 0, 0, 0]
DX = [0, 0, -1, 1, 0]
NAMES = ["上", "下", "左", "右", "停止"]


class _SafeUnpickler(pickle.Unpickler):
    """Q テーブル（dict と numpy 配列）だけを読む。それ以外のクラスは読み込まない。"""
    OK = {"builtins", "collections", "copyreg", "_codecs"}

    def find_class(self, module, name):
        if module in self.OK or module.startswith("numpy"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"読み込みを許可していない型です: {module}.{name}")


def load_qtables(path, ckpt_step=None):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        obj = _SafeUnpickler(f).load()
    if isinstance(obj, dict):                 # {step: [Qテーブル, ...]} の形（save の path_ckpt）
        step = ckpt_step if ckpt_step is not None else max(obj)
        print(f"チェックポイント {sorted(obj)} のうち step={step} を使います")
        obj = obj[step]
    if not isinstance(obj, (list, tuple)) or not all(isinstance(q, dict) for q in obj):
        raise SystemExit("Q テーブルの形が想定と違います（エージェントごとの dict のリストを想定）")
    return list(obj)


def guess(path, a):
    """ファイル名から手法・環境・N・λ を読み取る（指定があればそちらを優先）"""
    name = os.path.basename(path)
    if a.method is None:
        if "acthistory" in name:
            a.method = "prop"
        elif re.search(r"N=?None", name):
            a.method = "obs"
        else:
            a.method = "prior"
    if a.env is None:
        m = re.search(r"env=?(\d+)", name)
        a.env = int(m.group(1)) if m else 3
    if a.past_n is None and a.method != "obs":
        m = re.search(r"N=?(\d+)", name)
        a.past_n = int(m.group(1)) if m else 2
    if a.lam is None:
        m = re.search(r"lam=?([\d.]+\d)", name)
        a.lam = float(m.group(1)) if m else 0.9
    if a.method == "obs":
        a.past_n = None


def fmt_v(v):
    return "なし" if v is None else "(" + ", ".join(f"{float(x):g}" for x in v) + ")"


def parse_list(text, n, conv, what):
    items = [s.strip() for s in text.split(";")]
    if len(items) != n:
        raise SystemExit(f"{what} はエージェント数（{n}）と同じ数だけ ; で区切って指定してください: {text}")
    return [conv(s) for s in items]


def set_start(env, a, A):
    """初期位置・荷物・行動履歴をセットする"""
    env.reset()                               # まず学習開始時と同じ状態にしてから上書きする
    if a.pos:
        pos = parse_list(a.pos, A, lambda s: tuple(int(v) for v in s.split(",")), "--pos")
        for (y, x) in pos:
            if not (0 <= y < len(env.layout) and 0 <= x < len(env.layout[0])) or env.layout[y][x] != 1:
                raise SystemExit(f"({y},{x}) は通路ではありません")
        if len(set(pos)) != A:
            raise SystemExit("同じマスに複数のエージェントは置けません")
        env.agents_pos = list(pos)
    if a.cargo:
        cargo = [bool(int(v)) for v in a.cargo.split(",")]
        if len(cargo) != A:
            raise SystemExit(f"--cargo はエージェント数（{A}）と同じ数だけ指定してください")
        env.agents_cargo = cargo

    hist = parse_list(a.hist, A, lambda s: [ACT_CHAR[c] for c in s.upper() if c.strip()], "--hist") \
        if a.hist else [[] for _ in range(A)]

    if hasattr(env, "agents_actqueue"):       # 本手法（行動履歴ベクトル）の環境
        for i in range(A):
            env.agents_actqueue[i].clear()
            env.agents_relative_vector[i] = np.array([0, 0])
            for act in hist[i]:               # 学習中の step と同じ手順で履歴を積む
                env.update_relvec(i, True, act)
                env.agents_actqueue[i].append(act)
    elif hasattr(env, "agents_posqueue"):     # 先行研究（位置の差分）の環境
        for i in range(A):
            y, x = env.agents_pos[i]
            past = [(y, x)]
            for act in reversed(hist[i]):     # 今の位置から行動を逆にたどって、過去の位置を作る
                y, x = y - DY[act], x - DX[act]
                past.append((y, x))
            env.agents_posqueue[i].clear()
            for p in reversed(past):
                env.agents_posqueue[i].append(p)
    elif any(hist):
        print("★この環境には行動履歴がないので --hist は使いません")


def run_play(qtable, pos=None, cargo=None, hist=None, steps=2000, seed=None, temp=0.52,
             method=None, env=None, past_n=None, lam=None, ckpt_step=None, verbose=True):
    """学習済み Q テーブルで指定配置から動かし、軌跡（_log.npz の traj0 と同じ形の dict）を返す。学習はしない。"""
    a = argparse.Namespace(qtable=qtable, pos=pos, cargo=cargo, hist=hist, steps=steps, seed=seed, temp=temp,
                           method=method, env=env, past_n=past_n, lam=lam, ckpt_step=ckpt_step)
    qtables = load_qtables(a.qtable, a.ckpt_step)
    A = len(qtables)
    guess(a.qtable, a)
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)

    if a.method == "prop":
        from gridworld.environment_momentum import Environment
        envobj = Environment(A, a.env, a.past_n, lam=a.lam)
    else:
        from gridworld.environment import Environment
        envobj = Environment(A, a.env, a.past_n)
    if verbose:
        print(f"手法={a.method}  環境={a.env}  エージェント={A}体  N={a.past_n}  λ={a.lam if a.method == 'prop' else '-'}"
              f"  温度={a.temp}  シード={a.seed if a.seed is not None else '指定なし'}")

    agents = []
    for q in qtables:
        ag = BaseAgent()
        ag.q_table = q
        agents.append(ag)
    known = [set(q.keys()) for q in qtables]  # 学習中に訪れた状態

    set_start(envobj, a, A)
    if verbose:
        print("スタート: " + "  ".join(f"[{i}] ({y},{x}) 荷物{'あり' if c else 'なし'} v={fmt_v(envobj.get_relvec(i))}"
                                    for i, ((y, x), c) in enumerate(zip(envobj.agents_pos, envobj.agents_cargo))))

    T = a.steps
    tr = {"pos": np.zeros((T, A, 2), np.int8), "cargo": np.zeros((T, A), np.int8),
          "vec": np.full((T, A, 2), np.nan, np.float32), "q": np.zeros((T, A, 5), np.float32),
          "action": np.zeros((T, A), np.int8), "deliv": np.zeros((T, A), np.int8),
          "unseen": np.zeros((T, A), np.int8)}
    for t in range(T):
        for i, ag in enumerate(agents):       # 学習時と同じく 0, 1, 2 の順に1体ずつ動く
            ag.obs = envobj.get_obs(i)
            ag.cargo = envobj.agents_cargo[i]
            rel_vec = envobj.get_relvec(i)
            key = ag.convert_state(ag.obs, ag.cargo, rel_vec)
            seen = key in known[i]
            act = ag.get_action(a.temp, rel_vec)   # Q テーブルは参照するだけ（更新しない）

            tr["pos"][t, i] = envobj.agents_pos[i]
            tr["cargo"][t, i] = bool(ag.cargo)
            if rel_vec is not None:
                tr["vec"][t, i] = rel_vec[:2]   # 2成分だけ記録（run_randtag の番号付きでも動くように）
            tr["q"][t, i] = ag.q_table[key]
            tr["unseen"][t, i] = 0 if seen else 1

            reward, delivered, movable = envobj.step(i, act)
            tr["action"][t, i] = act if movable else 4
            tr["deliv"][t, i] = bool(delivered)
    tr["_meta"] = {"method": a.method, "temp": a.temp, "seed": a.seed, "qtable": os.path.basename(a.qtable)}
    return tr


def summarize(tr):
    """搬出回数・最初に動いたステップ・未訪問の状態にいたステップ数"""
    A = tr["pos"].shape[1]
    out = []
    for i in range(A):
        moved = np.flatnonzero(tr["action"][:, i] != 4)
        out.append({"deliv": int(tr["deliv"][:, i].sum()), "first_move": int(moved[0]) if moved.size else None,
                    "unseen": int(tr["unseen"][:, i].sum())})
    return out


def main():
    ap = argparse.ArgumentParser(description="学習済み Q テーブルで指定配置から動かす（学習なし）")
    ap.add_argument("qtable", help="_qtable_step=….pkl.gz")
    ap.add_argument("--pos", default=None)
    ap.add_argument("--cargo", default=None)
    ap.add_argument("--hist", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.52)
    ap.add_argument("--method", choices=["obs", "prior", "prop"], default=None)
    ap.add_argument("--env", type=int, default=None)
    ap.add_argument("--past_n", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--ckpt_step", type=int, default=None, help="複数のステップが入ったファイルのとき、使うステップ")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tr = run_play(a.qtable, a.pos, a.cargo, a.hist, a.steps, a.seed, a.temp,
                  a.method, a.env, a.past_n, a.lam, a.ckpt_step)
    meta = tr.pop("_meta")
    T, A = tr["pos"].shape[:2]
    out = a.out or re.sub(r"\.pkl(\.gz)?$", "", a.qtable) + "_play.npz"
    arrays = {f"traj0_{k}": v for k, v in tr.items()}
    arrays.update({"traj0_start": np.asarray(0), "traj0_end": np.asarray(T),
                   "meta_qtable": np.asarray(meta["qtable"]), "meta_method": np.asarray(meta["method"]),
                   "meta_temp": np.asarray(a.temp), "meta_seed": np.asarray(-1 if a.seed is None else a.seed)})
    np.savez_compressed(out, **arrays)

    print(f"\n{T:,} ステップ動かしました（学習なし）")
    for i, s in enumerate(summarize(tr)):
        fm = "一度も動かず" if s["first_move"] is None else f"最初に動いたのは {s['first_move']} ステップ目"
        print(f"  エージェント{i}: 搬出 {s['deliv']} 回、{fm}、未訪問の状態 {s['unseen']} ステップ")
    q0 = tr["q"][0]
    print("スタート時の Q 値（上 下 左 右 停止）:")
    for i in range(A):
        print(f"  エージェント{i}: " + "  ".join(f"{NAMES[k]} {q0[i, k]:.3g}" for k in range(5)))
    print(f"\n保存: {out}\n描画: python replay_pygame.py {out}")


if __name__ == "__main__":
    main()
