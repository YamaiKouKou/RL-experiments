"""学習済み Q テーブルを動かして（学習はしない）、渋滞・すれ違い・鉢合わせを手法ごとに比べる検証スクリプト。

置き場所: main_parallel.py と同じフォルダ（play_qtable.py と replay_pygame.py を使う）。env3（3体）前提。

使い方
  python conflict_tests.py <テスト名> Qテーブル1 Qテーブル2 ... [オプション]
  例) python conflict_tests.py approach q/*N3_relative_seed[0-3]_* q/*acthistory*_seed[0-3]_*
      python conflict_tests.py all q/*.pkl.gz --out result      # 全部やって result_*.csv に保存

テスト名（何を測るか）
  jam       搬入口前の渋滞テスト。(4,4)荷なし (4,5)荷なし (4,6)荷あり の一列から、
            荷物ありが搬出するまでのステップ数。役の割り当て6通り×乱数 --seeds 個、履歴は「停止→停止」「右→右」。
  sweep     ランダムな初期配置 --trials 個 × --steps ステップ。搬出数と「同じマスに居続けた最長ステップ」。
            初期配置と乱数は全 Q テーブルで共通なので、同じ配置どうしで比べられる。
  waste     sweep と同じ走行で、各ステップを「目的地（荷なし→搬入口, 荷あり→搬出口）に近づいたか」で判定し、
            近づかなかったステップ（停止・横・後退）＝ムダ歩を、場所（搬入口付近/合流部/搬出口付近）×
            「他のロボットが2マス以内にいたか」で分けて、搬出1回あたりで数える。
  cross     合流部すれ違いテスト。合流部（(2,3)(2,4)(3,3)(3,4)）に上り（荷あり）と下り（荷なし）を
            詰めて置き、3体全員が目的（上り→搬出 / 下り→荷物の受け取り）を果たすまでのステップ数。
            場面 S1/S2/S3 × 履歴「移動中」「停止中」× 役6通り × 乱数 --seeds 個。
  encounter sweep と同じ走行で、合流部の中で荷ありと荷なしが隣り合った「鉢合わせ」の回数と長さ（搬出100回あたり）。
  approach  sweep と同じ走行から、「鉢合わせになりかけた瞬間」を抜き出して、その後どうなったかを集計。
            なりかけ = 荷ありが下側（合流部の外、合流部まで2マス以内）、荷なしが上側（同じく2マス以内）に
                      同時にいる状態が始まった瞬間。2体とも合流部を通らないと目的地に行けない。
            集計: その後 --look ステップ以内に鉢合わせしたか / 2体とも通り抜けるまでのステップ数 /
                  その瞬間の各自の行動（進む・止まる・それ以外）/ どちらが先に合流部へ入ったか
  all       上の全部

オプション
  --seeds 40     jam / cross の乱数の数（役6通りそれぞれに）
  --trials 120   sweep 系のランダム初期配置の数
  --steps 300    sweep 系の1回のステップ数
  --look 8       approach で鉢合わせを見る先のステップ数
  --jobs N       並列数（既定: CPU 数）
  --out PREFIX   PREFIX_<テスト名>.csv に1行=1 Q テーブルで保存
"""
import argparse
import csv
import itertools
import os
import re
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import play_qtable as PQ                      # noqa: E402
import replay_pygame as RP                    # noqa: E402

# ---------------- 地図と距離 ----------------
LAYOUT = RP.LAYOUTS[3]
FREE = {(y, x) for y, row in enumerate(LAYOUT) for x, v in enumerate(row) if v == 1}
EXIT_CELL, INTAKE_CELL = (1, 1), (4, 6)       # ここに入ると搬出 / 荷物を受け取る
JUNCTION = {(2, 3), (2, 4), (3, 3), (3, 4)}   # 上の部屋と下の部屋をつなぐ幅2マスの通路
UPPER = {p for p in FREE if p[0] <= 2 and p not in JUNCTION}
LOWER = {p for p in FREE if p[0] >= 3 and p not in JUNCTION}


def bfs(sources):
    d = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        y, x = q.popleft()
        for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if n in FREE and n not in d:
                d[n] = d[(y, x)] + 1
                q.append(n)
    return d


DIST = {0: bfs([INTAKE_CELL]), 1: bfs([EXIT_CELL])}   # 荷物の有無 → 目的地までの距離
DJ = bfs(list(JUNCTION))                               # 合流部までの距離


def region(p):
    if DIST[1][p] <= 2:
        return "搬出口付近"
    if DIST[0][p] <= 2:
        return "搬入口付近"
    if p in JUNCTION:
        return "合流部"
    return "その他"


def label(path):
    n = os.path.basename(path)
    m = "本手法" if "acthistory" in n else ("観測のみ" if re.search(r"N=?None", n) else "先行研究")
    if "randtag" in n:
        m += "+ランダム番号"
    mm = re.search(r"_merge([A-Z]+)", n)
    if mm:
        m += f"（{mm.group(1)}で区別なし）"
    s = re.search(r"seed=?(\d+)", n)
    return f"{m} seed{s.group(1) if s else '?'}"


def play(q, pos=None, cargo=None, hist=None, steps=300, seed=0):
    tr = PQ.run_play(q, pos, cargo, hist, steps, seed, verbose=False)
    tr.pop("_meta", None)
    return tr


def fmt_pos(cells):
    return ";".join(f"{y},{x}" for y, x in cells)


def first_goal_time(tr, ag, loaded, cap):
    """荷ありなら最初の搬出、荷なしなら最初に荷物を受け取ったステップ（1始まり）。できなければ cap"""
    if loaded:
        d = np.flatnonzero(tr["deliv"][:, ag] > 0)
        return int(d[0]) + 1 if d.size else cap
    c = tr["cargo"][:, ag]
    f = np.flatnonzero((c[1:] == 1) & (c[:-1] == 0))
    return int(f[0]) + 1 if f.size else cap


def placed_runs(roles, hist_override, nseeds):
    """roles = [(位置, 荷物, 履歴)...] を役の割り当て6通りに展開して (pos, cargo, hist, 役→エージェント) を返す"""
    for perm in itertools.permutations(range(3)):
        pos, cg, hs = [None] * 3, [0] * 3, [None] * 3
        for r, ag in enumerate(perm):
            p, c, h = roles[r]
            pos[ag], cg[ag], hs[ag] = p, c, (hist_override or h)
        for s in range(nseeds):
            yield fmt_pos(pos), ",".join(map(str, cg)), ";".join(hs), perm, cg, s


# ---------------- 各テスト ----------------
def t_jam(q, a):
    roles = [((4, 4), 0, None), ((4, 5), 0, None), ((4, 6), 1, None)]
    row = {}
    for hname, h in (("停止→停止", "SS"), ("右→右", "RR")):
        ts = []
        for P, C, H, perm, cg, s in placed_runs(roles, h, a.seeds):
            tr = play(q, P, C, H, 300, s)
            ts.append(first_goal_time(tr, perm[2], True, 300))
        ts = np.array(ts)
        row[f"{hname} 中央値"] = float(np.median(ts))
        row[f"{hname} 平均"] = round(float(ts.mean()), 2)
        row[f"{hname} 300内に運べず"] = int((ts >= 300).sum())
    return row


CROSS = {
    "S1 上り2×下り1": [((3, 3), 1, "LL"), ((3, 4), 1, "LU"), ((2, 3), 0, "LD")],
    "S2 上り1×下り2": [((3, 3), 1, "LL"), ((2, 3), 0, "LD"), ((2, 4), 0, "RD")],
    "S3 縦列上り2×下り1": [((4, 4), 1, "LL"), ((3, 4), 1, "LU"), ((2, 4), 0, "RD")],
}


def t_cross(q, a, cap=100):
    row = {}
    for sn, roles in CROSS.items():
        for hv, ov in (("移動中", None), ("停止中", "SS")):
            ts = []
            for P, C, H, perm, cg, s in placed_runs(roles, ov, a.seeds):
                tr = play(q, P, C, H, cap, s)
                ts.append(max(first_goal_time(tr, ag, cg[ag] == 1, cap) for ag in range(3)))
            row[f"{sn} {hv}"] = round(float(np.mean(ts)), 2)
    return row


def sweep_runs(q, a):
    for s in range(a.trials):
        yield play(q, steps=a.steps, seed=s)


def longest_stay(tr):
    P = tr["pos"]
    best = 0
    for i in range(P.shape[1]):
        run = 0
        for t in range(1, len(P)):
            run = run + 1 if (P[t, i] == P[t - 1, i]).all() else 0
            best = max(best, run)
    return best


def is_encounter(P, C, t):
    for i in range(3):
        for j in range(3):
            if C[t, i] == 1 and C[t, j] == 0:
                p, r = tuple(P[t, i]), tuple(P[t, j])
                if p in JUNCTION and r in JUNCTION and abs(p[0] - r[0]) + abs(p[1] - r[1]) == 1:
                    return True
    return False


def t_sweep(q, a):
    d, lg = [], []
    for tr in sweep_runs(q, a):
        d.append(int(tr["deliv"].sum()))
        lg.append(longest_stay(tr))
    return {"搬出(平均)": round(float(np.mean(d)), 2), "居続け最長(平均)": round(float(np.mean(lg)), 2),
            "20step以上居続けた試行": int(sum(x >= 20 for x in lg)), "_deliv": d}


def t_waste(q, a):
    stat, deliv = {}, 0
    for tr in sweep_runs(q, a):
        P, C, DL = tr["pos"], tr["cargo"], tr["deliv"]
        deliv += int(DL.sum())
        for t in range(len(P) - 1):
            for i in range(3):
                if DL[t, i]:
                    continue
                p, c, p2 = tuple(P[t, i]), int(C[t, i]), tuple(P[t + 1, i])
                near = min(abs(p[0] - P[t, j][0]) + abs(p[1] - P[t, j][1]) for j in range(3) if j != i) <= 2
                prog = DIST[c][p2] < DIST[c][p] or (c == 0 and C[t + 1, i] == 1)
                k = f"{region(p)}・{'他あり' if near else '他なし'}"
                stat[k] = stat.get(k, 0) + (0 if prog else 1)
    row = {"ムダ歩/搬出(合計)": round(sum(stat.values()) / deliv, 3)}
    for r in ("搬入口付近", "合流部", "その他", "搬出口付近"):
        for n in ("他あり", "他なし"):
            row[f"{r}・{n}"] = round(stat.get(f"{r}・{n}", 0) / deliv, 3)
    return row


def t_encounter(q, a):
    deliv, steps, eps, lens = 0, 0, 0, []
    for tr in sweep_runs(q, a):
        P, C = tr["pos"], tr["cargo"]
        deliv += int(tr["deliv"].sum())
        run = 0
        for t in range(len(P)):
            if is_encounter(P, C, t):
                steps += 1
                run += 1
            elif run:
                eps += 1
                lens.append(run)
                run = 0
    return {"鉢合わせ回数/100搬出": round(100 * eps / deliv, 2), "1回の長さ": round(float(np.mean(lens)), 2) if lens else 0,
            "鉢合わせステップ/100搬出": round(100 * steps / deliv, 2)}


def act_kind(P, C, DL, t, i):
    """t の行動を「進む」（目的地に近づいた）「止まる」「それ以外」に分ける"""
    if DL[t, i]:
        return "進む"
    p, p2, c = tuple(P[t, i]), tuple(P[t + 1, i]), int(C[t, i])
    if p == p2:
        return "進む" if (c == 0 and C[t + 1, i] == 1) else "止まる"
    return "進む" if DIST[c][p2] < DIST[c][p] else "それ以外"


def t_approach(q, a, cap=20):
    ev = 0
    met = 0
    cross_t = []
    acts = {(w, k): 0 for w in ("荷あり", "荷なし") for k in ("進む", "止まる", "それ以外")}
    first_in = {"荷ありが先": 0, "荷なしが先": 0, "同時": 0, "どちらも入らず": 0}
    for tr in sweep_runs(q, a):
        P, C, DL = tr["pos"], tr["cargo"], tr["deliv"]
        T = len(P)
        prev = set()
        for t in range(T - 1):
            cur = set()
            for i in range(3):
                for j in range(3):
                    if C[t, i] == 1 and C[t, j] == 0:
                        pi, pj = tuple(P[t, i]), tuple(P[t, j])
                        if pi in LOWER and DJ[pi] <= 2 and pj in UPPER and DJ[pj] <= 2:
                            cur.add((i, j))
            for i, j in cur - prev:                   # なりかけが始まった瞬間だけ数える
                if t + cap >= T:
                    continue
                ev += 1
                acts[("荷あり", act_kind(P, C, DL, t, i))] += 1
                acts[("荷なし", act_kind(P, C, DL, t, j))] += 1
                # 鉢合わせしたか（この2体が合流部の中で隣り合う）
                hit = False
                for k in range(t, min(T, t + a.look + 1)):
                    pi, pj = tuple(P[k, i]), tuple(P[k, j])
                    if C[k, i] == 1 and C[k, j] == 0 and pi in JUNCTION and pj in JUNCTION \
                            and abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) == 1:
                        hit = True
                        break
                met += hit
                # 2体とも通り抜けるまで（荷あり: 上側に出る or 搬出 / 荷なし: 下側に出る or 受け取り）
                ti = tj = None
                for k in range(t, t + cap):
                    if ti is None and (DL[k, i] or tuple(P[k + 1, i]) in UPPER):
                        ti = k + 1 - t
                    if tj is None and (C[k + 1, j] == 1 or tuple(P[k + 1, j]) in LOWER):
                        tj = k + 1 - t
                cross_t.append(max(ti or cap, tj or cap))
                # どちらが先に合流部へ入ったか
                ei = next((k for k in range(t, t + cap) if tuple(P[k, i]) in JUNCTION), None)
                ej = next((k for k in range(t, t + cap) if tuple(P[k, j]) in JUNCTION), None)
                if ei is None and ej is None:
                    first_in["どちらも入らず"] += 1
                elif ej is None or (ei is not None and ei < ej):
                    first_in["荷ありが先"] += 1
                elif ei is None or ej < ei:
                    first_in["荷なしが先"] += 1
                else:
                    first_in["同時"] += 1
            prev = cur
    row = {"なりかけ回数": ev, f"{a.look}step以内に鉢合わせ(%)": round(100 * met / max(ev, 1), 1),
           "2体とも通り抜けるまで(平均)": round(float(np.mean(cross_t)), 2) if cross_t else 0}
    for (w, k), v in acts.items():
        row[f"{w}:{k}(%)"] = round(100 * v / max(ev, 1), 1)
    for k, v in first_in.items():
        row[f"先に合流部:{k}(%)"] = round(100 * v / max(ev, 1), 1)
    return row


TESTS = {"jam": t_jam, "sweep": t_sweep, "waste": t_waste, "cross": t_cross, "encounter": t_encounter,
         "approach": t_approach}


def _job(args):
    name, q, a = args
    return name, q, TESTS[name](q, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=list(TESTS) + ["all"])
    ap.add_argument("qtables", nargs="+")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--look", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = list(TESTS) if a.test == "all" else [a.test]
    jobs = [(n, q, a) for n in names for q in a.qtables]
    with Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(_job, jobs)
    for n in names:
        rows = [(q, r) for m, q, r in res if m == n]
        print(f"\n===== {n} =====")
        keys = [k for k in rows[0][1] if not k.startswith("_")]
        for k in keys:
            print(f"  {k}")
            print("    " + "  ".join(f"{label(q)}: {r[k]}" for q, r in rows))
        if n == "sweep" and len(rows) >= 2:             # 同じ配置どうしの比較（先行研究 vs 本手法、seed ごと）
            by = {label(q): np.array(r["_deliv"]) for q, r in rows}
            for lab, v in by.items():
                if lab.startswith("本手法"):
                    other = lab.replace("本手法", "先行研究")
                    if other in by:
                        d = v - by[other]
                        print(f"  同じ配置で比べて {lab} − {other}: 平均 {d.mean():+.2f}  本手法が多い {int((d > 0).sum())} / 少ない {int((d < 0).sum())}")
        if a.out:
            path = f"{a.out}_{n}.csv"
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                wr.writerow(["Qテーブル", "ラベル"] + keys)
                for q, r in rows:
                    wr.writerow([os.path.basename(q), label(q)] + [r[k] for k in keys])
            print(f"  保存: {path}")


if __name__ == "__main__":
    main()
