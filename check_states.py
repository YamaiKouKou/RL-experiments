"""Qテーブル(pickle)の状態数を、理論上の到達可能な状態空間と照合する。

使い方（src/ で実行）:
  python check_states.py --pkl 'result/<日時>/qtables_*seed0.pkl' --env 3 --agents 5 --past_n 2 --lam 0.9 --momentum
  python check_states.py --pkl q.pkl --env 3 --agents 1 --past_n None            # 観測のみ
  （--pkl は .pkl でも .pkl.gz でも可）
  (先行研究 = --momentum を付けない / 本手法 = --momentum と --lam)

 ・pickleは安全のため numpy配列と組み込み型しか復元できない読み込み方にしてある
 ・理論値は、あなたの Environment クラスをそのまま動かして求める（式の再実装による食い違いを避ける）
     1) 1体だけの環境で「到達可能な (位置, 荷物, 履歴ベクトル)」を全探索  … 履歴は他エージェントがいても同じ集合になる
        （ぶつかって動けない場合は「停止」と同じ扱いなので）
     2) 複数体なら、3×3の周囲8マスのうち通路のマスを他エージェントが占める全パターンを掛け合わせる
        → これを「理論上の上限」とする。Qテーブルにあって上限にない状態は、実装の食い違い（浮動小数点の誤差など）を意味する
 ・1体のときは上限＝正確な到達可能集合なので、通常 完全一致する
"""
import argparse, copy, glob, itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safeload import safe_load
from agent.base_agent import BaseAgent

def make_env(a, agents):
    if a.momentum:
        from gridworld.environment_momentum import Environment
        return Environment(agents, a.env, a.past_n, lam=a.lam)
    from gridworld.environment import Environment
    return Environment(agents, a.env, a.past_n)

def hist_sig(env):
    """履歴の中身（キューの内容）。同じ (位置,荷物,キュー) は同じ未来になる"""
    if getattr(env, "past_n", None) is None:
        return ()
    if hasattr(env, "agents_actqueue"):
        return ("a",) + tuple(env.agents_actqueue[0])
    return ("p",) + tuple(env.agents_posqueue[0])

def first_states(env, walk):
    out = []
    for cell in walk:                       # reset() 直後: 任意の通路セル・荷物なし・履歴なし
        e = copy.deepcopy(env)
        e.reset()
        e.agents_pos = [cell]
        if hasattr(e, "agents_posqueue") and e.past_n is not None:
            e.agents_posqueue[0].clear(); e.agents_posqueue[0].append(cell)
        out.append(e)
    return out

def private_states(a):
    """1体環境での到達可能な状態キー（周囲8マスは他エージェントなし）と、キーを作る元の (位置, 荷物, v)"""
    env = make_env(a, 1)
    walk = [(y, x) for y in range(len(env.layout)) for x in range(len(env.layout[0])) if env.layout[y][x] == 1]
    seen, stack, priv = set(), first_states(env, walk), {}
    while stack:
        e = stack.pop()
        sig = (e.agents_pos[0], e.agents_cargo[0], hist_sig(e))
        if sig in seen:
            continue
        seen.add(sig)
        priv[sig] = (e.agents_pos[0], e.agents_cargo[0], e.get_relvec(0))
        for act in range(5):
            e2 = copy.deepcopy(e)
            spawns = [None]
            # 搬出で終わるときだけ、リスポーン先ごとに分岐（random_spawn を固定して全候補を試す）
            orig = e2.random_spawn
            e2.random_spawn = lambda aid, o=orig: o(aid)
            probe = copy.deepcopy(e)
            _, delivered, _ = probe.step(0, act)
            if delivered:
                cands = [c for c in walk if c != probe_pos_before_spawn(e, act)]
                spawns = cands
            for sp in spawns:
                e3 = copy.deepcopy(e)
                if sp is not None:
                    e3.random_spawn = lambda aid, sp=sp: sp
                e3.step(0, act)
                stack.append(e3)
    return priv, env

def probe_pos_before_spawn(e, act):
    """搬出時、random_spawn が除外する現在地（移動後の位置）"""
    e2 = copy.deepcopy(e)
    e2.random_spawn = lambda aid: e2.agents_pos[aid]       # 位置を変えずに返す＝移動後の位置が取れる
    e2.step(0, act)
    return e2.agents_pos[0]

def neighbours8(layout, cell):
    y, x = cell
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(layout) and 0 <= nx < len(layout[0]) and layout[ny][nx] == 1:
                out.append((ny, nx))
    return out

def upper_bound(priv, env, n_agents):
    layout = env.layout
    base = BaseAgent()
    keys = set()
    for (cell, cargo, v) in priv.values():
        nb = neighbours8(layout, cell)
        for r in range(0, min(n_agents - 1, len(nb)) + 1):
            for occ in itertools.combinations(nb, r):
                env.agents_pos = [cell] + list(occ)
                obs = env.get_obs(0)
                keys.add(base.convert_state(obs, cargo, v))
    return keys

def norm(k):
    """pickleから読んだキーを比較用にそろえる（numpy型→素のPython型）"""
    obs, cargo, v = k
    v = None if v is None else tuple(float(x) for x in v)
    return (tuple(int(o) for o in obs), bool(cargo), v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, nargs="+")
    ap.add_argument("--env", type=int, default=3)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--past_n", default="None")
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--momentum", action="store_true", help="本手法（environment_momentum）を使う。付けなければ先行研究/観測のみ")
    a = ap.parse_args()
    a.past_n = None if str(a.past_n) == "None" else int(a.past_n)

    priv, env = private_states(a)
    vset = sorted({p[2] for p in priv.values()}, key=lambda t: (t is None, t))
    ub = {norm(k) for k in upper_bound(priv, env, a.agents)}
    if a.momentum and a.past_n is not None:   # 環境クラスとは独立な計算: 直近N回の行動の組み合わせ(5^N通り)の重み付き和
        D = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]       # 上,下,左,右,停止 (dy,dx)
        indep = {(round(sum(a.lam**k * D[c[k]][0] for k in range(a.past_n)), 6) + 0.0,
                  round(sum(a.lam**k * D[c[k]][1] for k in range(a.past_n)), 6) + 0.0)
                 for c in itertools.product(range(5), repeat=a.past_n)}
        print(f"[独立計算] 行動 {5**a.past_n} 通りの重み付き和の種類: {len(indep)}"
              f"（環境クラスから求めた v の種類 {len({p[2] for p in priv.values()})} と一致するはず）")
    print(f"[理論] 1体で到達可能な (位置,荷物,v): {len(priv)} 通り / v の種類: {len(vset)} / 周囲パターンまで含めた上限: {len(ub)} 状態")

    files = sorted({f for p in a.pkl for f in glob.glob(p)})
    if not files: raise SystemExit("pklが見つかりません")
    for f in files:
        tables = safe_load(f)
        print(f"\n== {f}  (エージェント{len(tables)}体)")
        union = set()
        for i, t in enumerate(tables):
            ks = {norm(k) for k in t}
            union |= ks
            out = ks - ub
            vs = {k[2] for k in ks}
            print(f"  agent{i}: Q表の状態 {len(ks):>7,} / 上限 {len(ub):,}  ({len(ks)/len(ub):.1%} を訪問)  v の種類 {len(vs)}/{len(vset)}  上限外 {len(out)}")
            if out:
                bad_v = sorted({k[2] for k in out if k[2] not in set(vset)}, key=str)
                print(f"    ★上限外の状態があります。うち v が理論集合にない値: {len(bad_v)} 種類 例: {bad_v[:5]}")
        if len(tables) > 1:
            print(f"  全エージェントの合算（重複除く）: {len(union):,} / 上限 {len(ub):,}")
        outside = union - ub
        miss = ub - union
        print(f"  判定: 上限外 {len(outside)}  → {'理論と矛盾なし' if not outside else '★実装と理論が食い違っています（浮動小数点の誤差・リセット漏れなどを疑う）'}")
        print(f"  未訪問 {len(miss)}  （理論上は到達できるが、学習後の方策では通らない組み合わせ。"
              f"履歴なしの小さな状態空間では 0 になり、履歴ありでは数個〜残るのが普通）")
        if 0 < len(miss) <= 12 and a.agents == 1:
            for k in sorted(miss, key=str): print("     ", k)


if __name__ == "__main__":
    main()
