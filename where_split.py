"""先行研究の1つの状態の中で、「直前3手の並び」によってその後の結果（目標値）がどれだけ違うかを測る。
＝ 行動を区別すると Q 値がどれだけ変わりうるか（区別が学習に作用しうる量）と、それがどこで大きいか。

使うもの: main_parallel.py を --traj_full と --upd_until 付きで回した _log.npz と _updates.npz。

計算
  1. 更新記録の各行（ある状態で、ある行動をとった判断 1 回）について、
       先行研究の状態 = (エージェント, 観測, 荷物, 直前3手の変位) と、実際の直前3手の並び（動きの記録から）を求める。
       ※ 本手法の記録でも同じ「先行研究の状態」にまとめて数える。
  2. (先行研究の状態, 行動) ごとに、並びごとの目標値の平均を出し、全体の平均とのずれを取る。
       区別したときの値の違い = √( Σ 回数×(並びの平均 − 全体の平均)² / 総回数 )
       説明できる割合(η²)     = 並びの違いで説明できるばらつき / 目標値の全体のばらつき
     先行研究はこの違いを1つの値に平均してしまう。本手法は並びごとに別の値を持てる。
  3. それを、行動（停止 / 移動）× 場所（マス、荷物の有無）ごとに集計し、大きい例も出す。

使い方
  python where_split.py 結果フォルダ... [--start 150000] [--end 1900000] [--out where]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ACT = ["上", "下", "左", "右", "停"]
DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


def label(path):
    n = os.path.basename(path)
    m = "本手法" if "acthistory" in n else ("観測のみ" if re.search(r"N=?None", n) else "先行研究")
    s = re.search(r"seed=?(\d+)", n)
    return m, int(s.group(1)) if s else -1


def load(lp, start, end):
    up = lp[:-len("_log.npz")] + "_updates.npz"
    lg, u = np.load(lp), np.load(up)
    A, D, P = lg["traj0_action"], lg["traj0_deliv"], lg["traj0_pos"]
    t = u["upd_act_step"].astype(np.int64)
    sel = (t >= start) & (t < min(end, len(A)))
    t = t[sel]
    ag = u["upd_agent"][sel].astype(np.int64)
    h = np.stack([A[np.maximum(t - k, 0), ag] for k in (3, 2, 1)], 1).astype(np.int64)   # 古い順
    fresh = (t < 3) | np.stack([D[np.maximum(t - k, 0), ag] for k in (3, 2, 1)], 1).any(1)
    dy = np.zeros(len(t), np.int64)
    dx = np.zeros(len(t), np.int64)
    for k in range(3):
        dy += np.vectorize(lambda a: DIR[a][0])(h[:, k]) if len(t) else 0
        dx += np.vectorize(lambda a: DIR[a][1])(h[:, k]) if len(t) else 0
    dy[fresh] = 0
    dx[fresh] = 0
    seq = h[:, 0] * 25 + h[:, 1] * 5 + h[:, 2]
    seq[fresh] = -1                                     # 搬出・開始直後（3手たまっていない）
    obs = u["upd_obs"][sel].astype(np.int64)
    okey = (obs * (5 ** np.arange(8))).sum(1)
    df = pd.DataFrame({
        "agent": ag, "okey": okey, "cargo": u["upd_cargo"][sel].astype(np.int64), "dy": dy, "dx": dx,
        "action": u["upd_action"][sel].astype(np.int64), "seq": seq,
        "target": u["upd_target"][sel].astype(float),
        "y": P[t, ag, 0].astype(np.int64), "x": P[t, ag, 1].astype(np.int64)})
    return df


def seq_name(s):
    if s < 0:
        return "搬出直後"
    return "→".join(ACT[(s // 25) % 5]) if False else f"{ACT[s // 25]}→{ACT[(s // 5) % 5]}→{ACT[s % 5]}"


def analyze(df):
    g = ["agent", "okey", "cargo", "dy", "dx", "action"]
    df = df.copy()
    df["gmean"] = df.groupby(g)["target"].transform("mean")
    sm = df.groupby(g + ["seq"]).agg(n=("target", "size"), smean=("target", "mean"), gmean=("gmean", "first"),
                                      y=("y", lambda v: v.mode().iat[0]), x=("x", lambda v: v.mode().iat[0])).reset_index()
    sm["ssb"] = sm["n"] * (sm["smean"] - sm["gmean"]) ** 2
    df["sst"] = (df["target"] - df["gmean"]) ** 2
    return df, sm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--start", type=int, default=150_000)
    ap.add_argument("--end", type=int, default=1_900_000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default="where")
    a = ap.parse_args()
    summary, cells = [], []
    for d in a.dirs:
        for lp in sorted(glob.glob(os.path.join(d, "*_log.npz"))):
            if not os.path.exists(lp[:-len("_log.npz")] + "_updates.npz"):
                continue
            m, s = label(lp)
            df, sm = analyze(load(lp, a.start, a.end))
            print(f"\n===== {m} seed{s}（{a.start:,}〜{a.end:,} ステップ、判断 {len(df):,} 回）=====")
            print("  行動    区別したときの値の違い(√平均二乗)   並びで説明できる割合(η²)")
            for nm, mask_d, mask_s in (("停止", df.action == 4, sm.action == 4), ("移動", df.action != 4, sm.action != 4)):
                N = mask_d.sum()
                ssb, sst = sm.loc[mask_s, "ssb"].sum(), df.loc[mask_d, "sst"].sum()
                print(f"  {nm}    {np.sqrt(ssb / N):6.2f}                          {100 * ssb / sst:5.1f}%")
                summary.append([m, s, nm, int(N), round(float(np.sqrt(ssb / N)), 3), round(float(100 * ssb / sst), 2)])
            # 場所ごと
            sm["kind"] = np.where(sm.action == 4, "停止", "移動")
            df["kind"] = np.where(df.action == 4, "停止", "移動")
            c1 = sm.groupby(["y", "x", "cargo", "kind"]).agg(ssb=("ssb", "sum"), n=("n", "sum")).reset_index()
            for _, r in c1.iterrows():
                cells.append([m, s, int(r.y), int(r.x), int(r.cargo), r.kind, int(r.n), float(np.sqrt(r.ssb / r.n))])
            # 大きい例（移動）: 状態×行動ごとに ssb が大きいもの
            g = ["agent", "okey", "cargo", "dy", "dx", "action"]
            top = sm[sm.action != 4].groupby(g)["ssb"].sum().sort_values(ascending=False).head(a.top)
            print("  ― 並びで結果が大きく違う例（移動）―")
            for key in top.index:
                rows = sm.set_index(g).loc[[key]].sort_values("n", ascending=False).head(4)
                where = f"({int(rows.y.iloc[0])},{int(rows.x.iloc[0])}) 荷物{'あり' if key[2] else 'なし'} 行動{ACT[key[5]]}"
                desc = "  ".join(f"[{seq_name(int(sq))}] {sm_:.1f}（{int(n)}回）" for sq, sm_, n in
                                 zip(rows.seq, rows.smean, rows.n))
                print(f"   {where}: 全体の平均 {rows.gmean.iloc[0]:.1f} ← {desc}")
    pd.DataFrame(summary, columns=["手法", "seed", "行動", "判断の回数", "区別したときの値の違い", "η²(%)"]).to_csv(
        a.out + "_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cells, columns=["手法", "seed", "y", "x", "荷物", "行動", "回数", "値の違い"]).to_csv(
        a.out + "_cells.csv", index=False, encoding="utf-8-sig")
    print(f"\n保存: {a.out}_summary.csv  {a.out}_cells.csv")


if __name__ == "__main__":
    main()
