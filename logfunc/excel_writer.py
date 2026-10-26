"""qpsp_log.py が記録した区間（序盤・中盤・終盤など）を Excel に書き出す。

シート = 区間 × エージェント（例: 序盤_agent0）。1 行 = 1 ステップ。
列: step, y, x, 荷物, v_x, v_y, Q_上, Q_下, Q_左, Q_右, Q_停止, 行動, 搬出
（座標・荷物・v・Q 値は行動する直前のもの。搬出=1 の次の行は出現位置に飛ぶ）
"""
from openpyxl import Workbook

ACT = ["上", "下", "左", "右", "停止"]
HEADER = ["step", "y", "x", "荷物", "v_x", "v_y"] + [f"Q_{a}" for a in ACT] + ["行動", "搬出"]


def write_traj_excel(path, traj, labels=None):
    """traj: QPSPLogAlgorithm.traj（区間ごとの dict のリスト）"""
    labels = labels or [f"区間{i}" for i in range(len(traj))]
    wb = Workbook()
    wb.remove(wb.active)
    for label, tr in zip(labels, traj):
        start = tr["start"]
        pos, cargo, vec, q, act, dl = (tr[k] for k in ("pos", "cargo", "vec", "q", "action", "deliv"))
        for i in range(pos.shape[1]):
            ws = wb.create_sheet(f"{label}_agent{i}"[:31])
            ws.append(HEADER)
            for t in range(pos.shape[0]):
                ws.append([start + t, int(pos[t, i, 0]), int(pos[t, i, 1]), int(cargo[t, i]),
                           round(float(vec[t, i, 0]), 6), round(float(vec[t, i, 1]), 6)]
                          + [round(float(x), 6) for x in q[t, i]]
                          + [ACT[int(act[t, i])], int(dl[t, i])])
            ws.freeze_panes = "A2"
    wb.save(path)
