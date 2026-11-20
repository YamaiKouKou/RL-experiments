"""--traj_full で記録した長い軌跡（npz）を、見やすい Excel に分けて書き出す。

Excel は 1 シート 1,048,576 行が上限で、100万行のシートは開くのも重い。
そこで「--chunk ステップごとに 1 ファイル」に分けて書き出す（既定は 10万ステップ = 1 ファイル）。

1 ファイルの中身
  まとめ    : 1 行 = 1 ステップ。全エージェントの座標・荷物・行動・搬出を横に並べたもの
              （エージェント同士の位置関係を見るとき用）
  agent0〜  : 1 行 = 1 ステップ。今までの _traj.xlsx と同じ列（v と 5 行動の Q 値つき）

使い方
  # 全部（10万ステップごとに 10 ファイル）
  python traj_to_excel.py ..._log.npz

  # 見たいところだけ（30万〜32万ステップ）
  python traj_to_excel.py ..._log.npz --start 300000 --end 320000

  # 1 ファイルを 5万ステップにする / 出力先を変える
  python traj_to_excel.py ..._log.npz --chunk 50000 --outdir excel_seed0

座標・荷物・v・Q 値は行動する直前のもの。搬出=1 の次の行は出現位置に飛ぶ。
"""
import argparse
import os
import time

import numpy as np
from openpyxl import Workbook

ACT = ["上", "下", "左", "右", "停止"]
HEADER = ["step", "y", "x", "荷物", "v_y(上下)", "v_x(左右)"] + [f"Q_{a}" for a in ACT] + ["行動", "搬出"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="main_parallel.py が保存した _log.npz")
    ap.add_argument("--window", type=int, default=0, help="何番目の記録区間か（--traj_full なら 0）")
    ap.add_argument("--start", type=int, default=None, help="書き出す最初のステップ（既定: 記録の最初）")
    ap.add_argument("--end", type=int, default=None, help="書き出す最後のステップ（含まない。既定: 記録の最後）")
    ap.add_argument("--chunk", type=int, default=100_000, help="1 ファイルに入れるステップ数")
    ap.add_argument("--no_q", action="store_true", help="agent シートを作らない（まとめシートだけ。速い）")
    ap.add_argument("--outdir", default=None, help="出力先フォルダ（既定: npz と同じ場所に <名前>_excel）")
    a = ap.parse_args()

    if a.chunk > 1_048_575:
        raise SystemExit("--chunk は 1,048,575 以下にしてください（Excel の行数の上限）")

    z = np.load(a.npz, allow_pickle=False)
    p = f"traj{a.window}_"
    if p + "pos" not in z.files:
        raise SystemExit(f"区間 {a.window} は記録されていません。ある区間: "
                         f"{sorted(k[:-6] for k in z.files if k.endswith('_start'))}")
    rec0 = int(z[p + "start"])
    pos, cargo, vec, q, act, dl = (z[p + k] for k in ("pos", "cargo", "vec", "q", "action", "deliv"))
    L, A = pos.shape[0], pos.shape[1]
    rec1 = rec0 + L

    s0 = rec0 if a.start is None else max(a.start, rec0)
    s1 = rec1 if a.end is None else min(a.end, rec1)
    if s0 >= s1:
        raise SystemExit(f"指定の範囲に記録がありません（記録は {rec0:,}〜{rec1:,} ステップ）")

    outdir = a.outdir or os.path.splitext(a.npz)[0].replace("_log", "") + "_excel"
    os.makedirs(outdir, exist_ok=True)
    print(f"記録 {rec0:,}〜{rec1:,} ステップ、{A} 体 → {s0:,}〜{s1:,} を {a.chunk:,} ステップごとに書き出します")

    sum_header = ["step"]
    for i in range(A):
        sum_header += [f"y{i}", f"x{i}", f"荷物{i}", f"行動{i}", f"搬出{i}"]

    for c0 in range(s0, s1, a.chunk):
        c1 = min(c0 + a.chunk, s1)
        t0 = time.time()
        i0, i1 = c0 - rec0, c1 - rec0
        # 先に Python の値に直しておく（セルごとに numpy から変換するより速い）
        P = pos[i0:i1].tolist(); C = cargo[i0:i1].tolist(); D = dl[i0:i1].tolist()
        AC = act[i0:i1].tolist()
        V = np.round(vec[i0:i1].astype(np.float64), 6).tolist()
        Q = None if a.no_q else np.round(q[i0:i1].astype(np.float64), 6).tolist()

        wb = Workbook(write_only=True)    # 大きいファイル用の書き方（メモリをあまり使わない）
        ws = wb.create_sheet("まとめ")
        ws.freeze_panes = "B2"
        ws.append(sum_header)
        for t in range(i1 - i0):
            row = [c0 + t]
            for i in range(A):
                row += [P[t][i][0], P[t][i][1], C[t][i], ACT[AC[t][i]], D[t][i]]
            ws.append(row)
        if not a.no_q:
            for i in range(A):
                ws = wb.create_sheet(f"agent{i}")
                ws.freeze_panes = "A2"
                ws.append(HEADER)
                for t in range(i1 - i0):
                    ws.append([c0 + t, P[t][i][0], P[t][i][1], C[t][i], V[t][i][0], V[t][i][1],
                               *Q[t][i], ACT[AC[t][i]], D[t][i]])
        name = os.path.join(outdir, f"step{c0:07d}-{c1:07d}.xlsx")
        wb.save(name)
        print(f"  {name}  （{time.time() - t0:.0f} 秒）")
    print("完了")


if __name__ == "__main__":
    main()
