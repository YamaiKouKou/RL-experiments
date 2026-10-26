"""実験の実行。設定はコマンドライン引数で変える（既定値は今までの定数と同じ）。

例:
  python main_parallel.py --method obs   --seeds 0-29
  python main_parallel.py --method prior --seeds 0-29
  python main_parallel.py --method prop  --lam 0.9 --seeds 0-9,20,21
  python main_parallel.py --method prop  --lam 1.0 --agents 1 --max_steps 1000000

記録・保存（既定: 200万ステップごとに Q テーブルを pickle.gz、シード0〜2 の序盤・中盤・終盤を Excel）
  --ckpt_every 2000000    : Q テーブルを pickle で保存する間隔（0 で保存しない。最後は常に保存）
  --traj_seeds 0-2        : 序盤・中盤・終盤の動きと Q 値を記録するシード（none で記録しない）
  --traj_len 2000         : 各区間のステップ数
  --no_excel              : 区間の記録を Excel にしない（npz だけ）

--method で手法を選ぶ:
  obs   : 観測のみ           (PAST_N=None, environment.py)
  prior : 先行研究(相対ベクトル) (environment.py)
  prop  : 本手法(行動履歴)     (environment_momentum.py, --lam を使う)
"""
import os
import argparse
import random
import subprocess
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.base_agent import BaseAgent
from algorithm.qpsp_log import QPSPLogAlgorithm
from logfunc.callback import LoggerCallback


def parse_seeds(text):
    """'0-29' や '0-9,20,21' をシード番号のリストにする。'none' なら空。"""
    if text.strip().lower() in ("", "none"):
        return []
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return sorted(set(seeds))


def get_args():
    ap = argparse.ArgumentParser(description="MARL 荷物搬送の実験")
    ap.add_argument("--method", choices=["obs", "prior", "prop"], required=True)
    ap.add_argument("--env", type=int, default=3)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--past_n", type=int, default=2, help="obs のときは無視（None になる）")
    ap.add_argument("--lam", type=float, default=0.9, help="prop のときだけ使う")
    ap.add_argument("--alpha", type=float, default=0.06)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--d_rate", type=float, default=0.20)
    ap.add_argument("--temp", type=float, default=0.52)
    ap.add_argument("--max_steps", type=int, default=5_000_000)
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0-9"), help="例: 0-29 / 0-9,20,21")
    ap.add_argument("--procs", type=int, default=os.cpu_count(), help="並列数（既定はCPUコア数）")
    ap.add_argument("--ckpt_every", type=int, default=2_000_000, help="Qテーブルを保存する間隔（0で途中保存なし）")
    ap.add_argument("--traj_seeds", type=parse_seeds, default=parse_seeds("0-2"), help="動きを記録するシード（none で無し）")
    ap.add_argument("--traj_len", type=int, default=2000, help="序盤・中盤・終盤それぞれのステップ数")
    ap.add_argument("--no_excel", action="store_true", help="区間の記録を Excel に書き出さない")
    ap.add_argument("--out", default="./result")
    a = ap.parse_args()
    if a.method == "obs":
        a.past_n = None
    return a


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def tag_of(cfg):
    """ファイル名に使う識別子。今までの名前（env=3_agent=5_N=2_relative）に λ を足した形。"""
    mode = "acthistory" if cfg["method"] == "prop" else "relative"
    tag = f"env={cfg['env']}_agent={cfg['agents']}_N={cfg['past_n']}_{mode}"
    if cfg["method"] == "prop":
        tag += f"_lam={cfg['lam']}"
    return tag


def run_simulation(job):
    """1シード分。Windows でも動くように、設定は引数で受け取る（グローバル変数に頼らない）。"""
    seed, cfg = job
    if cfg["method"] == "prop":
        from gridworld.environment_momentum import Environment
        env = Environment(cfg["agents"], cfg["env"], cfg["past_n"], lam=cfg["lam"])
    else:
        from gridworld.environment import Environment
        env = Environment(cfg["agents"], cfg["env"], cfg["past_n"])

    print(f"seed={seed} start", flush=True)
    random.seed(seed)
    np.random.seed(seed)

    tag = tag_of(cfg)
    base = os.path.join(cfg["outf"], f"{tag}_seed={seed}")
    log_file_path = base + ".txt"

    # 序盤・中盤・終盤の区間（このシードを記録する場合だけ）
    T, L = cfg["max_steps"], cfg["traj_len"]
    windows = None
    if seed in cfg["traj_seeds"] and L > 0:
        windows = [(0, L), (T // 2 - L // 2, T // 2 + (L - L // 2)), (T - L, T)]

    agents = [BaseAgent() for _ in range(cfg["agents"])]
    algo = QPSPLogAlgorithm(alpha=cfg["alpha"], gamma=cfg["gamma"], d_rate=cfg["d_rate"],
                            save_episodes=True,
                            ckpt_every=cfg["ckpt_every"] or None,
                            ckpt_path=base + "_qtable_step={step}.pkl.gz",
                            traj_windows=windows)
    callback = LoggerCallback(log_file_path, cfg["agents"], cfg["past_n"])
    callback.on_debug_log = lambda *args, **kw: None   # テキストのデバッグログは書かない

    history = algo.run(env, agents, cfg["temp"], cfg["max_steps"], callback=callback, block=cfg["block"])
    callback.on_seed_end(seed, history)

    # 最後の Q テーブル（途中保存と重なっていなければ）
    if not (cfg["ckpt_every"] and T % cfg["ckpt_every"] == 0):
        algo._checkpoint(T, agents)
    # ブロック・エピソード・区間の記録（npz）
    algo.save(base + "_log.npz",
              meta=dict(seed=seed, method=cfg["method"], lam=cfg["lam"] if cfg["method"] == "prop" else 1.0,
                        N=cfg["past_n"] if cfg["past_n"] else 0, git=cfg["git"]))
    # 区間の記録を Excel に
    if windows and not cfg["no_excel"]:
        from logfunc.excel_writer import write_traj_excel
        write_traj_excel(base + "_traj.xlsx", algo.traj, labels=["序盤", "中盤", "終盤"])
    print(f"seed={seed} done", flush=True)
    return seed, history


if __name__ == "__main__":
    a = get_args()
    cfg = vars(a).copy()
    cfg["git"] = git_hash()
    tag = tag_of(cfg)
    outf = os.path.join(a.out, f"{datetime.now():%Y%m%d-%H%M}_{tag}")
    os.makedirs(outf, exist_ok=True)
    cfg["outf"] = outf

    # 設定をそのまま残す（どのコード・どの設定で出した結果か後から分かるように）
    with open(os.path.join(outf, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    print("設定:", ", ".join(f"{k}={cfg[k]}" for k in
          ["method", "env", "agents", "past_n", "lam", "max_steps", "git"]), f"seeds={a.seeds}")

    with Pool(processes=min(a.procs, len(a.seeds))) as pool:
        results = pool.map(run_simulation, [(s, cfg) for s in a.seeds])
    results.sort(key=lambda x: x[0])

    # グラフ
    plt.figure()
    for seed, history in results:
        steps = [(i + 1) * a.block for i in range(len(history))]
        plt.plot(steps, history, label=f"seed={seed}")
    plt.xlabel("step"); plt.ylabel(f"deliveries per {a.block} steps")
    plt.title(tag); plt.grid(True, linestyle="--", alpha=0.6)
    if len(results) <= 15:
        plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(outf, f"{tag}.png"))

    # シードごとのログを1つにまとめる（plot_compare.py がそのまま読める形式）
    contiguous = a.seeds == list(range(a.seeds[0], a.seeds[-1] + 1))
    seed_str = f"{a.seeds[0]}-{a.seeds[-1]}" if contiguous else "_".join(map(str, a.seeds))
    combined = os.path.join(outf, f"{tag}_seed={seed_str}.txt")
    with open(combined, "w", encoding="utf-8") as cf:
        for seed in a.seeds:
            p = os.path.join(outf, f"{tag}_seed={seed}.txt")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    cf.write(f.read())
                os.remove(p)
    print("保存:", combined)
