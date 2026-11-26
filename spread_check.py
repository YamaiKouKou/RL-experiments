"""「書き込み」と「読み込み」のずれ ＝ 経験が別の場面に広がって使われている量 を、学習記録から測る。

使うもの: main_parallel.py を --traj_full と --upd_until 付きで回した _log.npz と _updates.npz（同じ seed）。

考え方
  停止を選ぶたびに、その判断はあとで Q 更新される（_updates.npz の 1 行）。その行には
    q_before : 判断した時点でその状態に書かれていた Q(停止)   … ロボットが「読んだ」点数
    target   : その場面で実際に止まった結果から計算した目標値 … その場面の「自分の経験」の点数
  がある。また _log.npz の動きの記録から、その判断の直前3手（実際の履歴）が分かる。
  実際の履歴ごとに、
    読んだ点数の平均 − 自分の経験の点数の平均 ＝ ずれ
  を出す。状態が履歴ごとに分かれていれば、Q 値は自分の経験の平均に近づくので、ずれはほぼ 0 になる。
  同じ状態に違う履歴が混ざっていると、Q 値は混ざった経験の平均になるので、
    ずれ > 0 … 他の履歴の（良い）経験で底上げされた点数を読んで止まっている
    ずれ < 0 … 他の履歴の（悪い）経験で押し下げられた点数を読んでいる
  になる。

使い方
  python spread_check.py 結果フォルダ... [--start 800000] [--end 2000000] [--out spread]
  フォルダの中の *_log.npz と、同じ名前の *_updates.npz を全部使う。--start〜--end は判断したステップの範囲。
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

CATS = ["停止→停止→停止", "搬出・開始直後", "直前に停止（それ以前に移動あり）", "直前に移動"]


def label(path):
    n = os.path.basename(path)
    m = "本手法" if "acthistory" in n else ("観測のみ" if re.search(r"N=?None", n) else "先行研究")
    s = re.search(r"seed=?(\d+)", n)
    return m, int(s.group(1)) if s else -1


def categories(A, D, t, ag):
    """判断したステップ t とエージェント ag の配列から、直前3手の履歴の種類（番号）を返す"""
    n = len(t)
    cat = np.full(n, 3, np.int8)                         # 既定: 直前に移動
    h = np.stack([A[np.maximum(t - k, 0), ag] for k in (3, 2, 1)], 1)       # 古い順
    dd = np.stack([D[np.maximum(t - k, 0), ag] for k in (3, 2, 1)], 1)
    cat[h[:, 2] == 4] = 2                               # 直前に停止
    cat[(h == 4).all(1)] = 0                            # 停止→停止→停止
    cat[(t < 3) | dd.any(1)] = 1                        # 搬出・開始直後（3手たまっていない）
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--start", type=int, default=800_000)
    ap.add_argument("--end", type=int, default=2_000_000)
    ap.add_argument("--out", default="spread")
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        for lp in sorted(glob.glob(os.path.join(d, "*_log.npz"))):
            up = lp[:-len("_log.npz")] + "_updates.npz"
            if not os.path.exists(up):
                continue
            m, s = label(lp)
            lg, u = np.load(lp), np.load(up)
            A, D = lg["traj0_action"], lg["traj0_deliv"]
            t = u["upd_act_step"].astype(np.int64)
            sel = (u["upd_action"] == 4) & (t >= a.start) & (t < min(a.end, len(A)))
            t = t[sel]
            ag = u["upd_agent"][sel].astype(np.int64)
            read = u["upd_q_before"][sel].astype(float)
            own = u["upd_target"][sel].astype(float)
            cat = categories(A, D, t, ag)
            n = len(t)
            print(f"\n{m} seed{s}: 停止の判断 {n:,} 回（{a.start:,}〜{a.end:,} ステップ）")
            print("   履歴                           割合    読んだ点数  自分の経験の点数    ずれ")
            tot_abs = 0.0
            for c, name in enumerate(CATS):
                k = cat == c
                if not k.any():
                    continue
                r_, o_ = read[k].mean(), own[k].mean()
                share = k.mean()
                tot_abs += share * abs(r_ - o_)
                rows.append([m, s, name, int(k.sum()), round(100 * share, 2), round(r_, 3), round(o_, 3), round(r_ - o_, 3)])
                print(f"   {name:24s} {100 * share:5.1f}%   {r_:8.2f}   {o_:10.2f}       {r_ - o_:+6.2f}")
            print(f"   ずれの大きさ（|ずれ| を割合で重み付けした平均）: {tot_abs:.2f}")
            rows.append([m, s, "ずれの大きさ（重み付き平均）", n, 100.0, "", "", round(tot_abs, 3)])
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["手法", "seed", "履歴", "停止の判断の回数", "割合(%)", "読んだ点数(Q停止)", "自分の経験の点数(目標値)", "ずれ"])
        w.writerows(rows)
    print("\n保存:", a.out + ".csv")


if __name__ == "__main__":
    main()
