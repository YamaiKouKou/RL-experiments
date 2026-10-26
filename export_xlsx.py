"""qpsp_log.py で保存したデータを、人が見る用の Excel に書き出す。

使い方
  # 軌跡: 区間0 の最初の 2000 ステップをエージェントごとのシートに
  python export_xlsx.py traj --npz run_seed=0.npz --window 0 --steps 2000 --out traj.xlsx

  # エピソード長: エージェントごとのシートに（多いときは --last で末尾だけ）
  python export_xlsx.py episodes --npz run_seed=0.npz --last 100000 --out episodes.xlsx

  # Q テーブル: 1 状態 1 行（観測 8 マス, 荷物, v, 5 行動の Q 値, 最大 Q, 最善行動）
  python export_xlsx.py qtable --pkl run_seed=0_qtable_step=5000000.pkl.gz --out qtable.xlsx   （.pkl も可）

Excel の 1 シートは 1,048,576 行が上限。超える分は書き出さずに警告する。
"""
import argparse
import gzip
import pickle
import numpy as np
from openpyxl import Workbook

ACT = ["上", "下", "左", "右", "停止"]
OBS_COLS = ["北西", "北", "北東", "西", "東", "南西", "南", "南東"]
ROW_LIMIT = 1_048_575  # 見出し行の分を引く


def new_book():
    wb = Workbook()
    wb.remove(wb.active)
    return wb


def add_sheet(wb, name, header, rows):
    ws = wb.create_sheet(name)
    ws.append(header)
    n = 0
    for r in rows:
        if n >= ROW_LIMIT:
            print(f"警告: シート {name} が行数の上限に達したので打ち切り")
            break
        ws.append(r)
        n += 1
    ws.freeze_panes = "A2"
    return n


def cmd_traj(a):
    z = np.load(a.npz)
    p = f"traj{a.window}_"
    if p + "pos" not in z.files:
        raise SystemExit(f"区間 {a.window} は保存されていません: {[f for f in z.files if f.endswith('_start')]}")
    start = int(z[p + "start"])
    pos, cargo, vec, act, dl = (z[p + k] for k in ("pos", "cargo", "vec", "action", "deliv"))
    T = pos.shape[0] if a.steps is None else min(a.steps, pos.shape[0])
    wb = new_book()
    header = ["step", "y", "x", "荷物", "v_x", "v_y", "行動", "搬出"]
    for i in range(pos.shape[1]):
        rows = ([start + t, int(pos[t, i, 0]), int(pos[t, i, 1]), int(cargo[t, i]),
                 round(float(vec[t, i, 0]), 6), round(float(vec[t, i, 1]), 6), ACT[act[t, i]], int(dl[t, i])]
                for t in range(T))
        add_sheet(wb, f"agent{i}", header, rows)
    wb.save(a.out)
    print(f"{a.out}: ステップ {start}〜{start + T - 1}, {pos.shape[1]} 体")


def cmd_episodes(a):
    z = np.load(a.npz)
    st, ag, ln, dv = z["ep_step"], z["ep_agent"], z["ep_length"], z["ep_delivered"]
    qs, qm = z["ep_q_start"], z["ep_q_max"]
    if a.last:
        st, ag, ln, dv, qs, qm = (x[-a.last:] for x in (st, ag, ln, dv, qs, qm))
    wb = new_book()
    header = ["終了step", "長さ", "搬出(1)/破棄(0)", "開始状態のmaxQ", "エピソード中のmaxQ"]
    for i in np.unique(ag):
        m = ag == i
        rows = zip(st[m].tolist(), ln[m].tolist(), dv[m].tolist(),
                   qs[m].astype(float).tolist(), qm[m].astype(float).tolist())
        n = add_sheet(wb, f"agent{int(i)}", header, rows)
        print(f"agent{int(i)}: {n} エピソード")
    wb.save(a.out)


def cmd_qtable(a):
    opener = gzip.open if a.pkl.endswith(".gz") else open   # .pkl.gz も .pkl も読める
    with opener(a.pkl, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):            # {ステップ: [各エージェントの Q テーブル]}
        keys = sorted(data)
        k = keys[-1] if a.ckpt == "last" else int(a.ckpt)
        tables = data[k]
        print(f"チェックポイント {k} ステップ（保存済み: {keys}）")
    else:                                  # [各エージェントの Q テーブル]
        tables = data
    wb = new_book()
    header = OBS_COLS + ["荷物", "v_x", "v_y"] + [f"Q_{x}" for x in ACT] + ["最大Q", "最善行動"]
    for i, qt in enumerate(tables):
        def rows():
            for (obs, cargo, v), q in qt.items():
                v = v if v is not None else (0, 0)
                q = np.asarray(q, float)
                yield list(map(int, obs)) + [int(bool(cargo)), float(v[0]), float(v[1])] + \
                    q.round(6).tolist() + [round(float(q.max()), 6), ACT[int(q.argmax())] if q.max() > 0 else "未学習"]
        n = add_sheet(wb, f"agent{i}", header, rows())
        print(f"agent{i}: {n} 状態")
    wb.save(a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("traj"); t.add_argument("--npz", required=True)
    t.add_argument("--window", type=int, default=0); t.add_argument("--steps", type=int)
    t.add_argument("--out", default="traj.xlsx")
    e = sp.add_parser("episodes"); e.add_argument("--npz", required=True)
    e.add_argument("--last", type=int); e.add_argument("--out", default="episodes.xlsx")
    q = sp.add_parser("qtable"); q.add_argument("--pkl", required=True)
    q.add_argument("--ckpt", default="last"); q.add_argument("--out", default="qtable.xlsx")
    a = ap.parse_args()
    {"traj": cmd_traj, "episodes": cmd_episodes, "qtable": cmd_qtable}[a.cmd](a)
