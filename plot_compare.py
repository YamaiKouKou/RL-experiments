"""3手法の搬出量を1枚に重ねる（シード平均線 ± S.D. の帯）。

main_parallel.py が出力する結合ログ（env=..._N=..._relative.txt / _acthistory.txt）を直接読む。
各ログには「[history のリスト]\nseed=k done」の形でシードごとの搬出量(1000ステップごと)が入っている。

使い方（src/ で実行）:
  python plot_compare.py \
    --obs   result/<観測のみ>/env=3_agent=5_N=None_relative.txt \
    --prior result/<先行研究>/env=3_agent=5_N=2_relative.txt \
    --prop  result/<本手法>/env=3_agent=5_N=2_acthistory.txt \
    --max_step 4000000 --out compare.png
"""
import argparse, ast, re
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "Yu Gothic", "Meiryo", "sans-serif"]

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
BLOCK = 1000  # main_parallel.py の BLOCK と同じ
STYLE = {"観測のみ": "#7f7f7f", "先行研究": "#ff7f0e", "本手法": "#1f77b4"}


def load(path):
    """結合ログから (シード数, ブロック数) の配列を返す。"""
    text = open(path, encoding="utf-8", errors="ignore").read()
    found = re.findall(r"^(\[[\d\., \-e]*\])\nseed=(\d+) done", text, re.M)
    if not found:
        raise SystemExit(f"history が見つかりません: {path}")
    by_seed = {int(s): ast.literal_eval(h) for h, s in found}  # 同じseedが複数あれば後勝ち
    T = min(len(v) for v in by_seed.values())
    return np.array([by_seed[s][:T] for s in sorted(by_seed)], dtype=float)


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.array([np.convolve(r, k, mode="valid") for r in x])


def report_tests(tails):
    """シード別の末尾平均（右の点グラフと同じ値）で、手法間をMann–Whitney U検定（両側）。
    シードごとに独立な学習なので対応なし。3組なのでHolm法で多重比較を補正。
    P(A>B)は「Aのランダムな1シードがBのランダムな1シードより大きい確率」（0.5で差なし、1.0で常にA>B）。"""
    from itertools import combinations
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("\n[検定] scipy が無いためスキップ（pip install scipy）"); return
    pairs = list(combinations(tails, 2)); rows = []
    for x, y in pairs:
        u = mannwhitneyu(tails[x], tails[y], alternative="two-sided")
        auc = u.statistic / (len(tails[x]) * len(tails[y]))
        rows.append((x, y, u.pvalue, auc))
    order = np.argsort([r[2] for r in rows]); m = len(rows); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):  # Holm補正
        run = max(run, min(1.0, rows[i][2] * (m - rank))); adj[i] = run
    print(f"\n[検定] Mann–Whitney U（両側・シード別の末尾平均, n={len(next(iter(tails.values())))}/手法）")
    print(f"{'A':<8}{'B':<8}{'P(A>B)':>8}{'p':>9}{'p(Holm)':>9}")
    for (x, y, p, auc), ph in zip(rows, adj):
        print(f"{x:<8}{y:<8}{auc:>8.2f}{p:>9.3f}{ph:>9.3f}" + ("  *" if ph < 0.05 else ""))
    print("  * = Holm補正後 p<0.05。シード10本では検出力が低く、差が無い＝同等ではない点に注意。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--prop", required=True)
    ap.add_argument("--window", type=int, default=20, help="移動平均の窓（ブロック数）")
    ap.add_argument("--tail", type=int, default=500, help="表の集計に使う末尾ブロック数（500=最後の50万ステップ）")
    ap.add_argument("--max_step", type=float, default=None, help="グラフの横軸の上限。未指定なら各手法を最後まで描く（表は常に各手法の全データの末尾で計算）")
    ap.add_argument("--ddof", type=int, default=1, help="S.D.の自由度（スライドの表と同じ値に）")
    ap.add_argument("--band", choices=["none", "minmax", "sd"], default="none",
                    help="本体グラフの帯: none=平均線のみ（既定）, minmax=最小〜最大, sd=平均±S.D.")
    ap.add_argument("--no_strip", action="store_true", help="右のシード別の点グラフを付けない")
    ap.add_argument("--out", default="compare.png")
    a = ap.parse_args()

    data = {"観測のみ": load(a.obs), "先行研究": load(a.prior), "本手法": load(a.prop)}
    if a.no_strip:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True); ax2 = None
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True,
                                      gridspec_kw={"width_ratios": [4.2, 1.5]})
    print(f"{'手法':<8}{'平均':>9}{'S.D.':>9}   (末尾{a.tail}ブロックの平均を、シード間で集計)")
    rng = np.random.default_rng(0)
    tails = {}
    for i, (name, full) in enumerate(data.items()):
        arr = full if not a.max_step else full[:, :int(a.max_step // BLOCK)]  # グラフ用
        sm = smooth(arr, a.window)
        mean, sd = sm.mean(0), sm.std(0, ddof=a.ddof)
        x = (np.arange(sm.shape[1]) + a.window) * BLOCK / 1e6  # 単位: 10^6 ステップ
        ax.plot(x, mean, color=STYLE[name], lw=2.6, label=name)
        if a.band != "none":
            lo, hi = (sm.min(0), sm.max(0)) if a.band == "minmax" else (mean - sd, mean + sd)
            ax.fill_between(x, lo, hi, color=STYLE[name], alpha=0.10, linewidth=0)
        # 集計区間: --max_step があれば左のグラフと同じ終点（全手法で同じ区間）、なければ各手法の最後
        end = min(full.shape[1], int(a.max_step // BLOCK)) if a.max_step else full.shape[1]
        if a.max_step and end < int(a.max_step // BLOCK):
            print(f"  ※ {name} は {full.shape[1]*BLOCK:,} ステップまでしかないため、その末尾で集計します")
        tail = full[:, end - a.tail:end].mean(1)
        tails[name] = tail
        ax.axvspan((end - a.tail) * BLOCK / 1e6, end * BLOCK / 1e6, color="#999", alpha=0.10, linewidth=0, zorder=0)
        print(f"{name:<8}{tail.mean():>9.2f}{tail.std(ddof=a.ddof):>9.2f}   seeds={full.shape[0]}  集計区間 {(end-a.tail)*BLOCK:,}〜{end*BLOCK:,} ステップ")
        if ax2 is not None:
            ax2.scatter(i + rng.uniform(-0.18, 0.18, len(tail)), tail, s=38, color=STYLE[name],
                        alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
            ax2.hlines(tail.mean(), i - 0.32, i + 0.32, color="#222", lw=2, zorder=4)
    ax.set_xlabel("ステップ数（×10⁶）", fontsize=16); ax.set_ylabel("搬出量（1000ステップあたり）", fontsize=16)
    ax.tick_params(labelsize=14); ax.grid(alpha=0.3); ax.margins(x=0.01)
    ax.legend(fontsize=15, loc="lower center", bbox_to_anchor=(0.5 if ax2 is None else 0.62, 1.0),
              ncol=3, frameon=False)
    if ax2 is not None:
        ax2.set_xlim(-0.6, len(data) - 0.4)
        ax2.set_xticks(range(len(data))); ax2.set_xticklabels(list(data), fontsize=11)
        ax2.set_xlabel("灰色の区間の\n● 1シード　― 平均", fontsize=11)
        ax2.tick_params(labelleft=False); ax2.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 330)
    report_tests(tails)
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
