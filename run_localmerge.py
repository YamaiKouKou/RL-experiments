"""本手法（行動履歴ベクトル）で、決めた場所・荷物のときだけ「行動の区別」をなくして学習する（局所的に先行研究と同じ粒度にする）。

区別をなくす場所では、状態の v の代わりに「直前 N 手の移動量の合計」（先行研究の状態と同じ情報、λ=1 の値）を使う。
それ以外の場所では、今まで通り本手法の v を使う。コードは書き換えず、
gridworld.environment_momentum.Environment.get_relvec を外から差し替える。
区別をなくした状態には印（3 つ目の要素 1）を付けて、他の場所の状態と Q 値の行が混ざらないようにしている。

場所の指定（--merge）
  A  : 荷物あり・搬入口から出口  (4,6) (3,6) (3,5) (3,4)
  B  : 荷物なし・下の通路と合流部 (4,3) (4,4) (4,5) (3,3) (2,3)
  AB : A と B の両方
  C  : A と B 以外の全部（荷物あり/なし、全マス）
  自分で決めるとき: --merge "4,6;3,6:1"  （マスを ; 区切り、: のあとに荷物 0/1。荷物を省くと両方）

使い方（main_parallel.py と同じフォルダに置く。--merge 以外の引数は main_parallel.py と同じ。--method prop で使う）
  python run_localmerge.py --merge A --method prop --agents 3 --past_n 3 --lam 0.9 --seeds 0-29 --max_steps 10000000
出力の名前には _mergeA（または _mergeB, _mergeX）が付く。

学習済み Q テーブルを動かして評価するときも、同じ差し替えが必要:
  python run_localmerge.py --merge A --eval conflict_tests.py sweep Qテーブル...
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridworld import environment_momentum as EM     # noqa: E402

A_CELLS = {(4, 6), (3, 6), (3, 5), (3, 4)}          # 荷物あり
B_CELLS = {(4, 3), (4, 4), (4, 5), (3, 3), (2, 3)}  # 荷物なし


def _in_a(pos, cargo):
    return cargo == 1 and pos in A_CELLS


def _in_b(pos, cargo):
    return cargo == 0 and pos in B_CELLS


PRESET = {                                          # (位置, 荷物) → 区別をなくすか
    "A": _in_a,
    "B": _in_b,
    "AB": lambda p, c: _in_a(p, c) or _in_b(p, c),
    "C": lambda p, c: not (_in_a(p, c) or _in_b(p, c)),
}
DY = [-1, 1, 0, 0, 0]
DX = [0, 0, -1, 1, 0]


def parse_merge(s):
    if s in PRESET:
        return PRESET[s], s
    cells_part, _, cargo_part = s.partition(":")
    cells = {tuple(int(v) for v in c.split(",")) for c in cells_part.split(";") if c.strip()}
    cargo = {int(cargo_part)} if cargo_part.strip() else {0, 1}
    return (lambda p, c: p in cells and c in cargo), "X"


def install(rule):
    orig = EM.Environment.get_relvec

    def get_relvec_merge(self, agent_id):
        v = orig(self, agent_id)
        if self.past_n is None:
            return v
        pos = tuple(int(x) for x in self.agents_pos[agent_id])
        if rule(pos, int(bool(self.agents_cargo[agent_id]))):
            q = self.agents_actqueue[agent_id]
            if len(q) < self.past_n:                   # 3手たまっていないときは元と同じく (0,0)
                return (0.0, 0.0, 1)
            return (float(sum(DY[a] for a in q)), float(sum(DX[a] for a in q)), 1)
        return v

    EM.Environment.get_relvec = get_relvec_merge


def main():
    argv = sys.argv[1:]
    if "--merge" not in argv:
        raise SystemExit("--merge A / B / \"y,x;y,x:荷物\" を指定してください")
    i = argv.index("--merge")
    spec = argv[i + 1]
    del argv[i:i + 2]
    rule, name = parse_merge(spec)
    install(rule)
    print(f"区別をなくす場所: {spec}")
    here = os.path.dirname(os.path.abspath(__file__))
    if argv and argv[0] == "--eval":
        sys.argv = argv[1:]
        runpy.run_path(argv[1], run_name="__main__")
        return
    sys.argv = [os.path.join(here, "main_parallel.py")] + argv
    g = runpy.run_path(os.path.join(here, "main_parallel.py"), run_name="__main__")
    outf, tag = g["outf"], g["tag"]
    suf = "_merge" + name
    for f in os.listdir(outf):
        if tag in f:
            os.rename(os.path.join(outf, f), os.path.join(outf, f.replace(tag, tag + suf)))
    new = outf.replace(tag, tag + suf)
    k = 2
    while os.path.exists(new):
        new = outf.replace(tag, tag + suf) + f"_{k}"
        k += 1
    os.rename(outf, new)
    print("保存先:", new)


if __name__ == "__main__":
    main()
