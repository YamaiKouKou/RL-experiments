"""Actions の結果(zip かそれを展開したフォルダ)をまとめて、要約表と1枚の図を作る。

使い方:
  python summarize_results.py result-obs-*.zip result-prior-*.zip result-prop-*.zip --out summary
  （展開済みフォルダでも可。手法は各フォルダの config.txt から自動で判定）

出力:
  summary.csv  … 手法ごとの要約（末尾平均・ばらつき・低い値で止まったシード数・収束までのステップなど）
  summary_1_delivery.png … 搬出量の推移 + シード別の末尾平均
  summary_2_maxQ.png     … エピソード開始状態の最大Q値
  summary_3_dQ.png       … Q値の変化量 |ΔQ|
  summary_4_states.png   … Qテーブルの状態数
  summary_5_eplen.png    … エピソード長（既定は搬出できたエピソードだけ。--eplen_all で打ち切りも含める）
  summary_6_discard.png  … 2000ステップで打ち切られたエピソードの数
  --slide を付けると、スライドに貼る用の大きな文字の図も出す
  summary_slide_curve.png … 搬出量の推移だけ（タイトルなし・文字大きめ）
  summary_slide_strip.png … シード別の末尾平均だけ（同上）
  summary_slide_maxQ.png  … エピソード開始時の最大Q値（同上）
  summary_slide_eplen.png … 1エピソードのステップ数（同上）
横軸はステップ数の対数（序盤の立ち上がりと、最後まで横ばいかを1枚で見るため）。
"""
import argparse, ast, csv, glob, logging, os, re, time, warnings, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "Yu Gothic", "Meiryo", "sans-serif"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NAME = {"obs": "観測のみ", "prior": "先行研究", "prop": "本手法"}
COLOR = {"obs": "#7f7f7f", "prior": "#ff7f0e", "prop": "#1f77b4"}
ORDER = ["obs", "prior", "prop"]


def find_runs(paths):
    """zip/フォルダから config.txt のあるフォルダを集める（zip は同じ場所に展開）"""
    runs = []
    expanded = []
    for p in paths:                       # Windows の PowerShell は * を展開しないので、ここで展開する
        hits = sorted(glob.glob(p))
        if not hits:
            raise SystemExit(f"見つかりません: {p}（今いるフォルダ: {os.getcwd()}）")
        expanded += hits
    for p in expanded:
        if p.endswith(".zip"):
            d = p[:-4]                    # zip と同じ名前のフォルダに展開する
            if os.path.isdir(d) and os.path.getmtime(p) > os.path.getmtime(d):
                # 同じ名前で新しい zip を落とし直した場合: 古い展開先は消さずに名前を変えて残し、展開し直す
                old = f"{d}_old_{time.strftime('%Y%m%d-%H%M%S')}"
                os.rename(d, old)
                print(f"zip の方が新しいので展開し直します（前の展開先は {old} に移しました）")
            if os.path.isdir(d):
                print(f"展開済みのフォルダを使います: {d}")
            else:
                print(f"展開します: {p} → {d}")
                with zipfile.ZipFile(p) as z:
                    z.extractall(d)
            p = d
        for cfg in glob.glob(os.path.join(p, "**", "config.txt"), recursive=True):
            runs.append(os.path.dirname(cfg))
    if not runs:
        raise SystemExit("config.txt のある結果フォルダが見つかりません")
    return runs


def read_config(d):
    cfg = {}
    for line in open(os.path.join(d, "config.txt"), encoding="utf-8"):
        if " = " in line:
            k, v = line.rstrip("\n").split(" = ", 1)
            cfg[k] = v
    return cfg


def load_run(d, eplen_all=False):
    cfg = read_config(d)
    txt = [f for f in glob.glob(os.path.join(d, "*.txt")) if not f.endswith("config.txt")][0]
    text = open(txt, encoding="utf-8").read()
    hist = {int(s): ast.literal_eval(h) for h, s in re.findall(r"^(\[[\d\., \-e]*\])\nseed=(\d+) done", text, re.M)}
    seeds = sorted(hist)
    T = min(len(hist[s]) for s in seeds)
    H = np.array([hist[s][:T] for s in seeds], float)
    blk = int(cfg.get("block", 1000))
    npz = {}
    for f in glob.glob(os.path.join(d, "*_log.npz")):
        s = int(re.search(r"seed=(\d+)_log\.npz$", f).group(1))
        npz[s] = f
    per = {"qstart": [], "dq": [], "states": [], "eplen": [], "discard": []}
    per_seeds = []                                        # per の各行がどのシードか
    for s in seeds:
        if s not in npz:
            continue
        per_seeds.append(s)
        z = np.load(npz[s])                               # allow_pickle=False（既定）で安全に読む
        nb = z["block_n_states"].shape[0]
        per["states"].append(z["block_n_states"].mean(1)[:T])
        per["dq"].append(np.nanmean(z["block_dq_mean"], 1)[:T])
        if "block_n_discard" in z.files:                                       # 古いログには無いことがある
            per["discard"].append(z["block_n_discard"].sum(1)[:T].astype(float))   # 全エージェント合計
        deliv = z["ep_delivered"].astype(bool) if "ep_delivered" in z.files else np.ones(len(z["ep_length"]), bool)
        L = z["ep_length"].astype(float)
        if not eplen_all:
            L = np.where(deliv, L, np.nan)          # 打ち切り（2000ステップ）は平均に入れない
        b = np.minimum(z["ep_step"] // blk, nb - 1)
        for key, val in (("qstart", z["ep_q_start"].astype(float)), ("eplen", L)):
            ok = np.isfinite(val)
            sm = np.bincount(b[ok], weights=val[ok], minlength=nb)
            ct = np.bincount(b[ok], minlength=nb)
            per[key].append(np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)[:T])
    if per["discard"] and len(per["discard"]) != len(per["states"]):
        per["discard"] = []                                                     # 一部のシードにしか無いなら使わない
    per = {k: np.array(v) for k, v in per.items()}
    per["seed"] = np.array(per_seeds, int)
    return cfg, seeds, H, blk, per


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w)
    v = np.where(np.isfinite(x), x, 0.0); n = np.isfinite(x).astype(float)
    return np.convolve(v, k, "same") / np.convolve(n, k, "same")


SHOW_BAND = False   # --band を付けたときだけ、シード間の 25〜75% の帯を描く


def band(ax, x, M, color, label=None, w=1):
    M = np.array([smooth(r, w) for r in M])
    mean = np.nanmean(M, 0)
    ax.plot(x, mean, color=color, lw=2, label=label)
    if SHOW_BAND:
        lo, hi = np.nanpercentile(M, [25, 75], 0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def label_log_y(ax):
    """対数の縦軸に 10, 20, 30, 50, 100, 200, ... の数値ラベルを付ける"""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# 同じ手法の結果を1つにまとめるとき、一致していないといけない設定
SAME_KEYS = ["env", "agents", "past_n", "lam", "alpha", "gamma", "d_rate", "temp", "block", "max_steps"]


def merge_runs(r1, r2):
    """同じ手法・同じ設定で、シードを分けて回した結果（例: seed=0-20 と seed=21-29）を1つにまとめる"""
    (c1, s1, H1, b1, p1), (c2, s2, H2, b2, p2) = r1, r2
    name = NAME[c1["method"]]
    diff = [k for k in SAME_KEYS if k in c1 and k in c2 and c1[k] != c2[k]]
    if diff or b1 != b2:
        raise SystemExit(f"{name}の結果をまとめられません。設定が違います: "
                         + ", ".join(f"{k}={c1.get(k)} / {c2.get(k)}" for k in diff or ["block"])
                         + f"\n  {c1['_dir']}\n  {c2['_dir']}")
    dup = sorted(set(s1) & set(s2))
    if dup:
        raise SystemExit(f"{name}の結果をまとめられません。同じシードが両方に入っています: {dup}\n"
                         f"  {c1['_dir']}\n  {c2['_dir']}\nどちらか一方だけを指定してください")
    if c1.get("git") != c2.get("git"):
        print(f"★注意: {name}の2つの結果はコードの版（git）が違います: {c1.get('git')} / {c2.get('git')}")
    T = min(H1.shape[1], H2.shape[1])
    if H1.shape[1] != H2.shape[1]:
        print(f"★注意: {name}の2つの結果はステップ数が違うので、短い方（{T * b1:,}ステップ）にそろえます")
    seeds = list(s1) + list(s2)
    order = np.argsort(seeds)
    H = np.vstack([H1[:, :T], H2[:, :T]])[order]
    per = {}
    for k in p1:
        a, b = p1[k], p2[k]
        if k == "seed":
            per[k] = np.concatenate([a, b])
        elif a.size and b.size:
            per[k] = np.vstack([a[:, :T], b[:, :T]])
        elif p1["seed"].size and p2["seed"].size:   # 片方のログにだけある項目は使わない（シードとずれるため）
            per[k] = np.array([])
        else:
            per[k] = a if a.size else b
    if per["seed"].size:                       # per もシード順に並べ直す
        o = np.argsort(per["seed"])
        per = {k: (v[o] if len(v) == len(o) else v) for k, v in per.items()}
    cfg = dict(c1)
    cfg["_dir"] = c1["_dir"] + " + " + c2["_dir"]
    return cfg, [seeds[i] for i in order], H, b1, per


def save_slide_figs(slide_data, a):
    """スライドに貼る用の図。文字を大きくし、タイトルは付けない（スライド側に書く）"""
    fs = {"tick": 15, "label": 17, "legend": 15}
    W, Hh = a.slide_size
    # 搬出量の推移
    f, ax = plt.subplots(figsize=(W, Hh), constrained_layout=True)
    for m, x, H, tail, per in slide_data:
        M = np.array([smooth(r, a.window) for r in H])
        ax.plot(x, np.nanmean(M, 0), color=COLOR[m], lw=2.5, label=NAME[m])
    ax.set_xscale("log")
    ax.set_xlabel("ステップ数（対数）", fontsize=fs["label"])
    ax.set_ylabel("搬出量（1000ステップあたり）", fontsize=fs["label"])
    ax.tick_params(labelsize=fs["tick"])
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=fs["legend"], loc="upper left", frameon=True)
    f.savefig(a.out + "_slide_curve.png", dpi=200)
    # シード別の末尾平均
    f, ax = plt.subplots(figsize=(W * 0.75, Hh), constrained_layout=True)
    for k, (m, x, H, tail, per) in enumerate(slide_data):
        ax.scatter(k + np.random.default_rng(0).uniform(-0.22, 0.22, len(tail)), tail, s=55,
                   color=COLOR[m], edgecolor="white", linewidth=0.6, zorder=3)
        ax.hlines(tail.mean(), k - 0.32, k + 0.32, color="#222", lw=2.5, zorder=4)
    ax.set_xticks(range(len(slide_data)))
    ax.set_xticklabels([NAME[m] for m, *_ in slide_data], fontsize=fs["tick"])
    ax.set_xlim(-0.6, len(slide_data) - 0.4)
    ax.set_ylabel("試行ごとの最終的な搬出量", fontsize=fs["label"])   # 最後の --tail ブロックの平均
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.grid(alpha=0.3, axis="y")
    f.savefig(a.out + "_slide_strip.png", dpi=200)
    # 最大Q値・エピソード長（ログ npz があるときだけ）
    for key, suffix, ylabel, logy in (("qstart", "maxQ", "エピソード開始時の最大Q値", False),
                                      ("eplen", "eplen", "1エピソードのステップ数", True)):
        if not all(per[key].size for *_, per in slide_data):
            continue
        f, ax = plt.subplots(figsize=(W, Hh), constrained_layout=True)
        for m, x, H, tail, per in slide_data:
            M = np.array([smooth(r, a.window) for r in per[key]])
            ax.plot(x[:M.shape[1]], np.nanmean(M, 0), color=COLOR[m], lw=2.5, label=NAME[m])
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log"); label_log_y(ax)
        ax.set_xlabel("ステップ数（対数）", fontsize=fs["label"])
        ax.set_ylabel(ylabel, fontsize=fs["label"])
        ax.tick_params(labelsize=fs["tick"])
        ax.grid(alpha=0.3, which="major")
        ax.legend(fontsize=fs["legend"], loc="best", frameon=True)
        f.savefig(a.out + f"_slide_{suffix}.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Actions の結果 zip か、展開したフォルダ")
    ap.add_argument("--tail", type=int, default=500, help="末尾平均に使うブロック数（500=最後の50万ステップ）")
    ap.add_argument("--low", type=float, default=0.95, help="末尾平均が「全シードの中央値×この値」未満なら『低い値で止まった』と数える")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--band", action="store_true", help="シード間の 25〜75パーセント の帯を描く（既定は平均の線だけ）")
    ap.add_argument("--exclude_stuck", action="store_true",
                    help="全停止したシード（末尾平均の搬出量が --stuck 未満）を除いてグラフと表を作る。出力名の末尾に _exstuck が付く")
    ap.add_argument("--stuck", type=float, default=1.0,
                    help="末尾平均の搬出量（1000ステップあたり、全エージェント合計）がこれ未満のシードを『全停止』とみなす")
    ap.add_argument("--eplen_all", action="store_true",
                    help="エピソード長に、2000ステップで打ち切られたエピソードも含める（既定は搬出できたものだけ）")
    ap.add_argument("--slide", action="store_true",
                    help="スライド用の図（文字を大きく、タイトルなし、1枚1グラフ）も出す: _slide_curve.png, _slide_strip.png")
    ap.add_argument("--slide_size", type=float, nargs=2, default=[6.4, 4.0], metavar=("幅", "高さ"),
                    help="スライド用の図の大きさ（インチ）。スライドに貼る大きさに近いほど文字が読みやすい")
    ap.add_argument("--out", default="summary")
    a = ap.parse_args()
    global SHOW_BAND
    SHOW_BAND = a.band
    if a.exclude_stuck:
        a.out += "_exstuck"                 # 除外なしの結果を上書きしないように

    runs = {}
    for d in find_runs(a.paths):
        cfg, seeds, H, blk, per = load_run(d, a.eplen_all)
        cfg["_dir"] = d
        print(f"読み込み: {NAME[cfg['method']]}  {len(seeds)}シード（{seeds[0]}〜{seeds[-1]}）  {H.shape[1]*blk:,}ステップ  "
              f"agents={cfg['agents']} N={cfg['past_n']} λ={cfg['lam'] if cfg['method']=='prop' else '-'}  git={cfg.get('git')}  [{d}]")
        m = cfg["method"]
        if m in runs:
            runs[m] = merge_runs(runs[m], (cfg, seeds, H, blk, per))
        else:
            runs[m] = (cfg, seeds, H, blk, per)
    for m, r in runs.items():
        if len(r[0]["_dir"].split(" + ")) > 1:
            print(f"→ {NAME[m]}: {len(r[0]['_dir'].split(' + '))}個の結果をまとめました  計{len(r[1])}シード  {r[2].shape[1]*r[3]:,}ステップ")
    agents = {r[0]["agents"] for r in runs.values()}
    if len(agents) > 1:
        print(f"★注意: エージェント数が手法ごとに違います {agents}（比べられません）")

    ms = [m for m in ORDER if m in runs]
    cond = f"環境{runs[ms[0]][0]['env']}・エージェント{', '.join(sorted(agents))}体"
    if a.exclude_stuck:
        cond += "（全停止シードを除外）"
    # 1枚目: 搬出量の推移 + シード別の末尾平均 / 2〜5枚目: 1枚ずつ
    f1, (aC, aS) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True,
                                gridspec_kw={"width_ratios": [3, 1.4]})
    figs = {k: plt.subplots(figsize=(8, 5), constrained_layout=True) for k in ("maxQ", "dQ", "states", "eplen", "discard")}
    aQ, aD, aN, aL, aX = (figs[k][1] for k in ("maxQ", "dQ", "states", "eplen", "discard"))

    rows = []
    seed_rows = []   # シードごとの値（グラフに使っている値をシード単位で）
    pos = 0
    drawn = []       # 右の図に並べた手法（全シード除外で描かなかった手法は入らない）
    slide_data = []  # スライド用の図に使う (手法, x, H, 末尾平均)
    for m in ms:
        cfg, seeds, H, blk, per = runs[m]
        T = H.shape[1]; x = (np.arange(T) + 1) * blk
        tail = H[:, -a.tail:].mean(1)
        # 全停止したシード（末尾の搬出量がほぼ 0）
        stuck = tail < a.stuck
        stuck_seeds = [s for s, k in zip(seeds, stuck) if k]
        n_all = len(seeds)
        if stuck_seeds:
            print(f"{NAME[m]}: 全停止したシード（末尾平均 < {a.stuck}）{len(stuck_seeds)}/{n_all} → {' '.join(map(str, stuck_seeds))}"
                  + ("　※除外します" if a.exclude_stuck else ""))
        if a.exclude_stuck:
            keep = ~stuck
            if not keep.any():
                print(f"★{NAME[m]}: 全シードが全停止しているので、この手法は描きません")
                continue
            seeds = [s for s, k in zip(seeds, keep) if k]
            H = H[keep]; tail = tail[keep]
            if per["seed"].size:
                pk = np.isin(per["seed"], seeds)
                per = {k: (v[pk] if len(v) == len(pk) else v) for k, v in per.items()}
        low_th = a.low * np.median(tail)
        n_low = int((tail < low_th).sum())
        # 収束までのステップ: 移動平均が、そのシード自身の末尾平均の98%に初めて達した時点
        t98 = []
        for r, tv in zip(H, tail):
            sm = np.convolve(r, np.ones(a.window) / a.window, "valid")
            t98.append((int(np.argmax(sm >= 0.98 * tv)) + a.window) * blk)
        row = {"手法": NAME[m], "シード数": len(seeds), "エージェント数": cfg["agents"], "総ステップ": T * blk,
               "全停止シード数": f"{len(stuck_seeds)}/{n_all}", "全停止シード": " ".join(map(str, stuck_seeds)),
               "除外": "全停止を除外" if a.exclude_stuck else "なし",
               "末尾平均": round(tail.mean(), 2), "S.D.": round(tail.std(ddof=1), 2),
               "中央値": round(float(np.median(tail)), 2), "最小": round(tail.min(), 1), "最大": round(tail.max(), 1),
               f"低い値で止まったシード数(<中央値×{a.low})": n_low,
               "低い値のシード": " ".join(str(s) for s, t in zip(seeds, tail) if t < low_th),
               "収束までのステップ(中央値)": int(np.median(t98)), "収束までのステップ(最大)": int(max(t98))}
        if per["states"].size:
            row["最終の状態数(1体あたり平均)"] = round(float(per["states"][:, -1].mean()), 1)
            row["状態数が最後に増えたステップ(中央値)"] = int(np.median(
                [(np.flatnonzero(np.diff(r) > 0)[-1] + 2) * blk if (np.diff(r) > 0).any() else blk for r in per["states"]]))
            row["末尾の|ΔQ|(中央値)"] = float(f"{np.nanmedian(per['dq'][:, -a.tail:]):.4g}")
            row["末尾の開始状態maxQ"] = round(float(np.nanmean(per["qstart"][:, -a.tail:])), 3)
            row["末尾のエピソード長"] = round(float(np.nanmean(per["eplen"][:, -a.tail:])), 2)
            if per["discard"].size:
                row["末尾の打ち切り数(1000ステップあたり)"] = round(float(np.nanmean(per["discard"][:, -a.tail:])), 3)
        rows.append(row)

        has = per["states"].size and len(per["states"]) == len(seeds)
        for i, sd in enumerate(seeds):
            sr = {"手法": NAME[m], "seed": sd, "搬出量(末尾平均)": round(float(tail[i]), 2),
                  "低い": "★" if tail[i] < low_th else "", "収束ステップ": t98[i]}
            if has:
                sr.update({"状態数(最後)": round(float(per["states"][i, -1]), 1),
                           "|ΔQ|(末尾)": float(f"{np.nanmean(per['dq'][i, -a.tail:]):.3g}"),
                           "開始maxQ(末尾)": round(float(np.nanmean(per["qstart"][i, -a.tail:])), 3),
                           "エピソード長(末尾)": round(float(np.nanmean(per["eplen"][i, -a.tail:])), 2)})
                if per["discard"].size:
                    sr["打ち切り数(末尾)"] = round(float(np.nanmean(per["discard"][i, -a.tail:])), 3)
            seed_rows.append(sr)

        c = COLOR[m]
        lab = f"{NAME[m]}（{len(seeds)}/{n_all}シード）" if a.exclude_stuck else f"{NAME[m]}（{len(seeds)}シード）"
        band(aC, x, H, c, lab, a.window)
        slide_data.append((m, x, H, tail, per))
        aS.scatter(pos + np.random.default_rng(0).uniform(-0.2, 0.2, len(tail)), tail, s=30, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)
        aS.hlines(tail.mean(), pos - 0.3, pos + 0.3, color="#222", lw=2, zorder=4)
        pos += 1
        drawn.append(m)
        if per["states"].size:
            band(aQ, x, per["qstart"], c, lab, a.window)
            band(aD, x, per["dq"], c, lab, a.window)
            band(aN, x, per["states"], c, lab, 1)
            band(aL, x, per["eplen"], c, lab, a.window)
            if per["discard"].size:
                band(aX, x, per["discard"], c, lab, a.window)

    for axx, t in [(aC, "搬出量（1000ステップあたり）"), (aQ, "エピソード開始状態の最大Q値"),
                   (aD, "Q値の変化量 |ΔQ|（Q更新1回あたり）"), (aN, "Qテーブルの状態数（1体あたり）"),
                   (aL, "エピソード長（搬出できたエピソード）" if not a.eplen_all else "エピソード長（打ち切りを含む）"),
                   (aX, "打ち切られたエピソード数（1000ステップあたり・全員合計）")]:
        axx.set_xscale("log"); axx.set_title(f"{t}　{cond}", loc="left", fontsize=12)
        axx.grid(alpha=0.3, which="both")
        axx.set_xlabel("ステップ数（対数）", fontsize=12)
        axx.legend(fontsize=10, loc="best")
    aD.set_yscale("log"); aL.set_yscale("log")
    label_log_y(aL)   # エピソード長の縦軸に数値を出す
    aS.set_xticks(range(len(drawn))); aS.set_xticklabels([NAME[m] for m in drawn], fontsize=11)
    aS.set_title(f"シード別の末尾平均（最後の{a.tail*1000:,}ステップ）\n● 1シード ― 平均"
                 + ("（全停止シードを除外）" if a.exclude_stuck else ""), loc="left", fontsize=12)
    aS.grid(alpha=0.3, axis="y")
    outs = [(f1, "_1_delivery")] + [(figs[k][0], n) for k, n in
                                    (("maxQ", "_2_maxQ"), ("dQ", "_3_dQ"), ("states", "_4_states"),
                                     ("eplen", "_5_eplen"), ("discard", "_6_discard"))]
    for f, suffix in outs:
        f.savefig(a.out + suffix + ".png", dpi=150)
    if a.slide and slide_data:
        save_slide_figs(slide_data, a)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:   # Excel で文字化けしないよう BOM 付き
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    skeys = list(dict.fromkeys(k for r in seed_rows for k in r))
    with open(a.out + "_seeds.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=skeys); w.writeheader(); w.writerows(seed_rows)

    # ---- コマンドラインに表示 ----
    def table(rs, cols):
        def width(t):  # 全角を2文字分として数える
            return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(t))
        ws = {c: max(width(c), *(width(r.get(c, "")) for r in rs)) for c in cols}
        pad = lambda t, c: " " * (ws[c] - width(t)) + str(t)
        print("  " + "  ".join(pad(c, c) for c in cols))
        for r in rs:
            print("  " + "  ".join(pad(r.get(c, ""), c) for c in cols))

    print(f"\n==== 手法ごとの要約（末尾＝最後の{a.tail * 1000:,}ステップ）====")
    table(rows, ["手法", "シード数", "全停止シード数", "末尾平均", "S.D.", "中央値", "最小", "最大",
                 f"低い値で止まったシード数(<中央値×{a.low})", "収束までのステップ(中央値)"])
    for r in rows:
        if r["低い値のシード"]:
            print(f"  {r['手法']}: 低い値で止まったシード → {r['低い値のシード']}")
    for m in ms:
        rs = [r for r in seed_rows if r["手法"] == NAME[m]]
        print(f"\n==== {NAME[m]}: シードごと（★＝低い値で止まったシード、搬出量の低い順）====")
        table(sorted(rs, key=lambda r: r["搬出量(末尾平均)"]), [c for c in skeys if c != "手法"])
    if a.slide:
        print(f"スライド用: {a.out}_slide_curve.png, _slide_strip.png, _slide_maxQ.png, _slide_eplen.png")
    print(f"\n保存: {a.out}_1_delivery.png ～ {a.out}_6_discard.png, {a.out}.csv（手法ごと）, {a.out}_seeds.csv（シードごと）")


if __name__ == "__main__":
    main()