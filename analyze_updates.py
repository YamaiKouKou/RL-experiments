"""Q 更新の記録（_updates.npz）を Excel にまとめて、停止の Q 値がどう育ったかを調べる。
updates_to_excel.py の機能もこれに入れた（updates_to_excel.py はもう使わなくてよい）。

■ 何が分かるか
  「学習の途中、搬出に成功したエピソードの中で停止をとっていた場面」を 1 つずつ並べる。
  例: 「搬出ステップ 412,035 のエピソード（長さ 23 ステップ）で、搬出の 5 手前に (4,4) で停止した。
       その停止に配られた報酬は r·d^5 = 0.00032、次の状態の Q 値から γ·maxQ = 9.1 が足され、
       Q(停止) は 8.7 → 8.8 になった。そのとき Q(上) は 2.1 しかなかった」

■ 使い方
  python analyze_updates.py <_updates.npz>  [オプション]

  よく使う例
    # 崩壊（全停止）する前までの更新を見る
    python analyze_updates.py result/xxx_seed=0_updates.npz --end 794000

    # 位置 (4,4) で停止した場面だけ、エージェント 0 だけ
    python analyze_updates.py result/xxx_seed=0_updates.npz --end 794000 --pos 4,4 --agent 0

    # 搬出の 5 手前以内の停止だけ（ゴール直前の停止）
    python analyze_updates.py result/xxx_seed=0_updates.npz --near 5

  オプション
    --start / --end : 搬出ステップの範囲（--end は含まない）
    --agent         : このエージェントだけ
    --pos y,x       : この位置でとった停止だけ（座標は _log.npz の軌跡から取る）
    --near K        : 搬出の K 手前以内の停止だけ
    --max_episodes  : 「エピソードの中身」に載せるエピソードの数（既定 300。Excel が重くならないように）
    --log           : 座標と Q 値を取る _log.npz（既定: 同じ名前の _log.npz）
    --out           : 保存先（既定: <npz の名前>_analysis.xlsx）

■ 必要なもの
  _updates.npz : main_parallel.py を --upd_until 付きで回すと出る（qpsp_updlog.py が書き出す）
  _log.npz     : 座標・その時の全行動の Q 値を出したいとき。--traj_full で軌跡を記録しておくこと
                 （無くても動くが、座標と「そのときの他の行動の Q 値」は空欄になる）

■ 出力される Excel のシート
  使い方          : 各シートと列の説明
  停止の記録      : 停止が更新された場面 1 つにつき 1 行
  エピソード一覧  : 搬出したエピソード 1 つにつき 1 行（長さ、停止の回数など）
  エピソードの中身: 停止を含むエピソードの全ステップ（古い順）。どの行動にいくら配られたか
  停止が高い状態  : 最後に Q(停止) が移動より大きくなっていた状態の一覧
  行動別まとめ    : エージェント×行動ごとの合計
"""
import argparse
import os
from collections import Counter

import numpy as np
from openpyxl import Workbook

ACT = ["上", "下", "左", "右", "停止"]
OBS = "北西,北,北東,西,東,南西,南,南東"
EXCEL_MAX = 1_048_000


def r6(v):
    v = float(v)
    return None if v != v else float(f"{v:.6g}")


def vtxt(y, x):
    return "なし" if (y != y or x != x) else f"({round(float(y), 6) + 0.0:g}, {round(float(x), 6) + 0.0:g})"


def load_updates(path, a):
    z = np.load(path, allow_pickle=False)
    U = {k[4:]: z[k] for k in z.files if k.startswith("upd_") and z[k].ndim >= 1}
    m = np.ones(len(U["deliv_step"]), bool)
    if a.start is not None:
        m &= U["deliv_step"] >= a.start
    if a.end is not None:
        m &= U["deliv_step"] < a.end
    if a.agent is not None:
        m &= U["agent"] == a.agent
    U = {k: v[m] for k, v in U.items()}
    # エージェント → 搬出ステップ → とった順（i が大きい＝古い行動が先）に並べる
    order = np.lexsort((-U["i"].astype(np.int64), U["deliv_step"], U["agent"]))
    U = {k: v[order] for k, v in U.items()}
    U["dq"] = (U["q_after"] - U["q_before"]).astype(np.float64)
    return U, float(z["upd_alpha"]), float(z["upd_gamma"]), float(z["upd_d_rate"])


def attach_traj(U, log_path):
    """行動したステップの座標と、そのときの全行動の Q 値を軌跡の記録から取る"""
    n = len(U["agent"])
    U["y"] = np.full(n, -1); U["x"] = np.full(n, -1); U["qall"] = np.full((n, 5), np.nan)
    if not log_path or not os.path.exists(log_path):
        return False
    z = np.load(log_path, allow_pickle=False)
    found = False
    steps = U["act_step"].astype(np.int64); ag = U["agent"].astype(np.int64)
    for k in z.files:
        if not (k.startswith("traj") and k.endswith("_pos")):
            continue
        w = k[:-4]; s0 = int(z[w + "_start"]); pos = z[k]; q = z[w + "_q"] if w + "_q" in z.files else None
        m = (steps >= s0) & (steps < s0 + pos.shape[0])
        if m.any():
            found = True
            U["y"][m] = pos[steps[m] - s0, ag[m], 0]; U["x"][m] = pos[steps[m] - s0, ag[m], 1]
            if q is not None:
                U["qall"][m] = q[steps[m] - s0, ag[m]]
    return found


def next_state_info(U):
    """各行の「次の状態」（同じエピソードで 1 つ後の行動）と比べて、観測が変わったか・状態が同じか"""
    n = len(U["agent"])
    same = np.full(n, -1, np.int8); chg = np.full(n, -1, np.int8)
    ep = U["deliv_step"].astype(np.int64) * 64 + U["agent"]
    # 並びはエピソードごとに古い順なので、次の状態は 1 つ下の行
    nxt = np.r_[ep[1:] == ep[:-1], False]
    j = np.flatnonzero(nxt)
    obs_same = np.all(U["obs"][j] == U["obs"][j + 1], axis=1)
    v_same = (np.isclose(U["vy"][j], U["vy"][j + 1]) | (np.isnan(U["vy"][j]) & np.isnan(U["vy"][j + 1]))) & \
             (np.isclose(U["vx"][j], U["vx"][j + 1]) | (np.isnan(U["vx"][j]) & np.isnan(U["vx"][j + 1])))
    c_same = U["cargo"][j] == U["cargo"][j + 1]
    same[j] = (obs_same & v_same & c_same).astype(np.int8)
    chg[j] = (~obs_same).astype(np.int8)
    return same, chg


def write_help(wb, a, U, alpha, gamma, d, have_traj):
    ws = wb.create_sheet("使い方")
    rows = [
        ["この Excel の見方"],
        [f"対象: 搬出ステップ {int(U['deliv_step'].min()):,}〜{int(U['deliv_step'].max()):,}"
         + (f"　エージェント {a.agent} のみ" if a.agent is not None else "")
         + (f"　位置 {a.pos} の停止のみ" if a.pos else "") + (f"　搬出の {a.near} 手前以内の停止のみ" if a.near else "")],
        [f"α={alpha}　γ={gamma}　d={d}　座標と全行動の Q 値: {'あり' if have_traj else 'なし（_log.npz に軌跡の記録がない）'}"],
        [],
        ["Q 値の更新は、搬出に成功したエピソードの中でだけ起きる（2000 ステップで打ち切られたエピソードは学習されない）。"],
        ["搬出した瞬間に、そのエピソードの行動を搬出に近い方から順に更新する: 新Q = 旧Q + α(目標値 − 旧Q)"],
        ["目標値 = 配られた報酬 r·d^i + γ·maxQ(次の状態)。i は搬出の何手前か（i=0 は搬出した行動で、目標値は r/(1−γ)）。"],
        [],
        ["シート", "内容"],
        ["停止の記録", "停止が更新された場面 1 つにつき 1 行。どのエピソードの何手前で、どこで止まり、いくら配られ、Q がどう変わったか"],
        ["エピソード一覧", "搬出したエピソード 1 つにつき 1 行。長さ、停止の回数、停止に配られた報酬の合計など"],
        ["エピソードの中身", f"停止を含むエピソードの全ステップを古い順に（最大 {a.max_episodes} エピソード）"],
        ["停止が高い状態", "範囲の最後の時点で Q(停止) が移動のどれよりも大きかった状態。差の大きい順"],
        ["行動別まとめ", "エージェント×行動ごとの更新回数・配られた報酬・ΔQ の合計"],
        [],
        ["列", "意味"],
        ["搬出step", "そのエピソードで搬出したステップ（＝更新が起きたステップ）"],
        ["行動step", "その行動をとったステップ（＝搬出step − i）"],
        ["何手前(i)", "搬出の何手前の行動か。0 が搬出した行動"],
        ["配られた報酬 r·d^i", "その行動に直接配られた報酬。d=0.2 なので数手前でほぼ 0 になる"],
        ["γ·maxQ(次)", "次の状態の最大 Q 値 × γ。Q 値の伸びの大部分はこちらから来る"],
        ["直後に観測が変化", "次のステップで周囲 3×3 の観測が変わったか（停止なら＝周りのエージェントが動いた）"],
        ["そのときのQ_上〜停止", "その行動をとる直前の、全行動の Q 値（軌跡の記録がある場合）"],
    ]
    for r in rows:
        ws.append(r)


def main():
    ap = argparse.ArgumentParser(description="Q 更新の記録を Excel にまとめる（停止の解析用）")
    ap.add_argument("npz", help="_updates.npz")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--agent", type=int, default=None)
    ap.add_argument("--pos", default=None, help="y,x この位置でとった停止だけ")
    ap.add_argument("--near", type=int, default=None, help="搬出の K 手前以内の停止だけ")
    ap.add_argument("--max_episodes", type=int, default=300)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    U, alpha, gamma, d = load_updates(a.npz, a)
    if len(U["agent"]) == 0:
        raise SystemExit("指定した範囲に更新がありません")
    log_path = a.log or (a.npz[:-len("_updates.npz")] + "_log.npz" if a.npz.endswith("_updates.npz") else None)
    have_traj = attach_traj(U, log_path)
    if a.pos and not have_traj:
        raise SystemExit("--pos を使うには、軌跡の記録（--traj_full）が入った _log.npz が必要です")
    same, chg = next_state_info(U)
    n = len(U["agent"])
    print(f"更新 {n:,} 行（搬出ステップ {int(U['deliv_step'].min()):,}〜{int(U['deliv_step'].max()):,}）"
          f"  座標: {'あり' if have_traj else 'なし'}")

    # ---- 注目する停止 ----
    stop = U["action"] == 4
    if a.near is not None:
        stop &= U["i"] <= a.near
    if a.pos:
        py, px = (int(v) for v in a.pos.split(","))
        stop &= (U["y"] == py) & (U["x"] == px)
    sidx = np.flatnonzero(stop)
    ep_key = U["deliv_step"].astype(np.int64) * 64 + U["agent"]

    # ---- エピソードごとの集計 ----
    edges = np.r_[0, np.flatnonzero(np.diff(ep_key)) + 1, n]
    ep_rows = []
    for f, e in zip(edges[:-1], edges[1:]):
        acts = U["action"][f:e]; st = acts == 4
        ii = U["i"][f:e][st]
        ep_rows.append([int(U["deliv_step"][f]), int(U["agent"][f]), int(U["ep_len"][f]), int(st.sum()),
                        int(ii.min()) if st.any() else None,
                        " ".join(str(int(v)) for v in sorted(ii)) if st.any() else "",
                        r6(U["r_alloc"][f:e][st].sum()) if st.any() else 0,
                        r6(U["dq"][f:e][st].sum()) if st.any() else 0,
                        int(stop[f:e].any())])

    # ---- 状態ごとの最後の Q（停止が高い状態） ----
    S = {}
    for j in range(n):
        key = (int(U["agent"][j]), tuple(U["obs"][j].tolist()), int(U["cargo"][j]), vtxt(U["vy"][j], U["vx"][j]))
        s = S.setdefault(key, {"q": [0.0] * 5, "n": [0] * 5, "pos": Counter()})
        s["q"][int(U["action"][j])] = float(U["q_after"][j]); s["n"][int(U["action"][j])] += 1
        if U["y"][j] >= 0:
            s["pos"][(int(U["y"][j]), int(U["x"][j]))] += 1
    high = sorted(((s["q"][4] - max(s["q"][:4]), k, s) for k, s in S.items()
                   if s["n"][4] and s["q"][4] > max(s["q"][:4])), key=lambda t: -t[0])

    # ---- 画面に要約 ----
    print(f"\n注目する停止: {len(sidx):,} 回（{len(set(ep_key[sidx].tolist())):,} エピソード）")
    if len(sidx):
        print(f"  搬出の何手前か: 中央値 {np.median(U['i'][sidx]):.0f}、"
              f"配られた報酬 r·d^i の平均 {U['r_alloc'][sidx].mean():.2e}、γ·maxQ(次) の平均 {U['next_term'][sidx].mean():.2f}")
        nx = chg[sidx] >= 0
        if nx.any():
            print(f"  停止の直後に観測が変わった（周りが動いた）割合: {(chg[sidx][nx] == 1).mean():.0%}")
        print("  最初の 5 件:")
        for j in sidx[:5]:
            p = f"({U['y'][j]},{U['x'][j]})" if U["y"][j] >= 0 else "-"
            print(f"    搬出step {int(U['deliv_step'][j]):,}  agent{int(U['agent'][j])}  長さ {int(U['ep_len'][j])}  "
                  f"{int(U['i'][j])}手前  位置{p}  報酬 {U['r_alloc'][j]:.2e}  γmaxQ(次) {U['next_term'][j]:.2f}  "
                  f"Q(停止) {U['q_before'][j]:.3f}→{U['q_after'][j]:.3f}")
    print(f"停止が移動より高い状態: {len(high):,}")

    # ---- Excel ----
    out = a.out or os.path.splitext(a.npz)[0] + "_analysis.xlsx"
    wb = Workbook(write_only=True)
    write_help(wb, a, U, alpha, gamma, d, have_traj)

    ws = wb.create_sheet("停止の記録")
    ws.freeze_panes = "A2"
    ws.append(["搬出step", "agent", "エピソード長", "何手前(i)", "行動step", "y", "x", "荷物", "v", f"観測({OBS})",
               "配られた報酬 r·d^i", "γ·maxQ(次)", "目標値", "更新前Q(停止)", "更新後Q(停止)", "ΔQ",
               "直後に観測が変化"] + [f"そのときのQ_{x}" for x in ACT])
    if len(sidx) > EXCEL_MAX:
        print(f"★停止の記録が多すぎるので、最初の {EXCEL_MAX:,} 行だけ Excel に書きます（--start/--end で範囲を絞ってください）")
    for j in sidx[:EXCEL_MAX]:
        ws.append([int(U["deliv_step"][j]), int(U["agent"][j]), int(U["ep_len"][j]), int(U["i"][j]),
                   int(U["act_step"][j]), int(U["y"][j]) if U["y"][j] >= 0 else None,
                   int(U["x"][j]) if U["x"][j] >= 0 else None, int(U["cargo"][j]), vtxt(U["vy"][j], U["vx"][j]),
                   ",".join(str(int(v)) for v in U["obs"][j]), r6(U["r_alloc"][j]), r6(U["next_term"][j]),
                   r6(U["target"][j]), r6(U["q_before"][j]), r6(U["q_after"][j]), r6(U["dq"][j]),
                   {1: "変化", 0: "同じ", -1: None}[int(chg[j])]] + [r6(v) for v in U["qall"][j]])

    ws = wb.create_sheet("エピソード一覧")
    ws.freeze_panes = "A2"
    ws.append(["搬出step", "agent", "エピソード長", "停止の回数", "最も搬出に近い停止(何手前)", "停止した何手前の一覧",
               "停止に配られた報酬の合計", "停止のΔQの合計", "注目する停止を含む"])
    for r in ep_rows[:EXCEL_MAX]:
        ws.append(r)

    ws = wb.create_sheet("エピソードの中身")
    ws.freeze_panes = "A2"
    ws.append(["搬出step", "agent", "エピソード長", "行動step", "何手前(i)", "y", "x", "荷物", "v", f"観測({OBS})", "行動",
               "配られた報酬 r·d^i", "γ·maxQ(次)", "目標値", "更新前Q", "更新後Q", "ΔQ", "直後に観測が変化", "注目する停止"]
              + [f"そのときのQ_{x}" for x in ACT])
    eps = list(dict.fromkeys(ep_key[sidx].tolist()))[:a.max_episodes]
    want = set(eps)
    for f, e in zip(edges[:-1], edges[1:]):
        if int(ep_key[f]) not in want:
            continue
        for j in range(f, e):
            ws.append([int(U["deliv_step"][j]), int(U["agent"][j]), int(U["ep_len"][j]), int(U["act_step"][j]),
                       int(U["i"][j]), int(U["y"][j]) if U["y"][j] >= 0 else None,
                       int(U["x"][j]) if U["x"][j] >= 0 else None, int(U["cargo"][j]),
                       vtxt(U["vy"][j], U["vx"][j]), ",".join(str(int(v)) for v in U["obs"][j]),
                       ACT[int(U["action"][j])], r6(U["r_alloc"][j]), r6(U["next_term"][j]), r6(U["target"][j]),
                       r6(U["q_before"][j]), r6(U["q_after"][j]), r6(U["dq"][j]),
                       {1: "変化", 0: "同じ", -1: None}[int(chg[j])], "★" if stop[j] else ""]
                      + [r6(v) for v in U["qall"][j]])
        ws.append([])

    ws = wb.create_sheet("停止が高い状態")
    ws.freeze_panes = "A2"
    ws.append(["Q(停止)−最大Q(移動)", "agent", "荷物", "v", f"観測({OBS})"] + [f"Q_{x}" for x in ACT]
              + [f"更新回数_{x}" for x in ACT] + ["主な位置"])
    for gap, (ag, obs, c, v), s in high:
        ws.append([r6(gap), ag, c, v, ",".join(map(str, obs))] + [r6(x) for x in s["q"]] + s["n"]
                  + [str(s["pos"].most_common(1)[0][0]) if s["pos"] else None])

    ws = wb.create_sheet("行動別まとめ")
    ws.append(["agent", "行動", "更新回数", "配られた報酬の合計", "ΔQの合計", "γ·maxQ(次)の平均", "直後に観測が変化した割合"])
    for ag in sorted(set(U["agent"].tolist())):
        for act in range(5):
            k = (U["agent"] == ag) & (U["action"] == act)
            if not k.any():
                ws.append([ag, ACT[act], 0]); continue
            nx = chg[k] >= 0
            ws.append([ag, ACT[act], int(k.sum()), r6(U["r_alloc"][k].sum()), r6(U["dq"][k].sum()),
                       r6(U["next_term"][k].mean()), r6((chg[k][nx] == 1).mean()) if nx.any() else None])
    wb.save(out)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
