import math
import random
from collections import deque

class Environment:
    def __init__(self, agent_num, env_num, past_n=None):
        #環境の2次元配列（0:壁，1：通路，2：他エージェント，3：搬入口，4:搬出口）
        if env_num == 1:
            #環境I
            self.layout = [[0,0,0,0,0,0],
                           [4,1,1,1,1,0],
                           [0,1,1,0,1,0],
                           [0,1,1,1,1,0],
                           [0,1,1,0,1,3],
                           [0,0,0,0,0,0]]
        elif env_num == 2:
            #環境II
            self.layout = [[0,0,0,0,0,0,0,0],
                           [4,1,1,1,1,0,0,0],
                           [0,1,1,1,1,1,1,0],
                           [0,1,1,1,1,1,1,0],
                           [0,0,0,1,1,1,1,3],
                           [0,0,0,0,0,0,0,0]]
        elif env_num == 3:
            #環境III
            self.layout = [[0,0,0,0,0,0,0,0],
                           [4,1,1,1,1,0,0,0],
                           [0,1,1,1,1,0,0,0],
                           [0,0,0,1,1,1,1,0],
                           [0,0,0,1,1,1,1,3],
                           [0,0,0,0,0,0,0,0]]
        else:
            #オリジナル環境(知覚的エイリアシング発生しそう)
            self.layout = [[3,1,1,1,1],
                           [0,0,0,0,1],
                           [0,0,0,0,1],
                           [4,1,1,1,1]]
        
        #エージェントの数
        self.agent_num = agent_num
        #各エージェントの位置
        self.agents_pos = []
        #各エージェントが荷物を持っているか管理
        self.agents_cargo = []

        #-----------------------------
        #各エージェントの相対ベクトル
        self.agents_relative_vector = [[0,0] for _ in range(self.agent_num)]
        #相対ベクトルは過去nステップを参照するか
        self.past_n = past_n
        if self.past_n is None:
            self.agents_posqueue = [deque(maxlen=1) for _ in range(self.agent_num)]
        #過去nステップ分の各エージェント座標履歴
        else:
            self.agents_posqueue = [deque(maxlen=self.past_n + 1) for _ in range(self.agent_num)]
        
    #エージェントの初期位置をランダムにセットし最初の観測情報を返す関数
    def reset(self):
        #エージェントの初期位置候補の2次元配列(通路のみ)
        agents_walkway = []
        for y in range(len(self.layout)):
            for x in range(len(self.layout[0])):
                if self.layout[y][x] == 1:
                    agents_walkway.append((y,x))
        #候補から重複なしで選ぶ
        self.agents_pos = random.sample(agents_walkway,self.agent_num)

        #全てのエージェントの荷物情報をリセット
        self.agents_cargo = [False for _ in range(self.agent_num)]

        #-----------------------------------
        #エージェント座標履歴をキューに追加
        for agent_id,pos in enumerate(self.agents_pos):
            self.agents_posqueue[agent_id].append(pos)
        
    #エージェントのランダムなリスポーン先選出
    def random_spawn(self,agent_id):
        agent_spawnable = []
        for y in range(len(self.layout)):
            for x in range(len(self.layout[0])):
                if self.layout[y][x] == 1 and (y,x) not in self.agents_pos:
                    agent_spawnable.append((y,x))

        return random.choice(agent_spawnable)
        
    #観測情報取得関数
    def get_obs(self,agent_id):
        #エージェントの現在座標取得
        ay, ax = self.agents_pos[agent_id]
        #観測情報初期化
        observation = [[0,0,0],
                       [0,0,0],
                       [0,0,0]]
        #周囲1マスに関して走査
        for i,dy in enumerate([-1,0,1]):
            for j,dx in enumerate([-1,0,1]):
                #自分がいる座標の時とばす
                if dy == 0 and dx == 0:
                    cell_value = 1
                else:
                    ny,nx = ay+dy, ax+dx
                    #境界線チェック
                    if 0 <= ny < len(self.layout) and 0 <= nx < len(self.layout[0]):
                        #レイアウト情報取得
                        cell_value = self.layout[ny][nx]
                        #エージェントがいるかどうかを判定,いる場合情報上書き
                        for pos in self.agents_pos:
                            if pos == (ny,nx):
                                cell_value = 2
                    else:
                        cell_value = 0
                #観測情報上書き
                observation[i][j] = cell_value
                
        return observation

    #------------------------------
    #相対ベクトルを取得する関数
    def get_relvec(self,agent_id):
        if self.past_n is None:
            return (0,0)
        
        if len(self.agents_posqueue[agent_id]) == self.agents_posqueue[agent_id].maxlen:
            cy,cx = self.agents_pos[agent_id]
            py,px = self.agents_posqueue[agent_id][0]

            rel_vec = (cy-py, cx-px)
            return rel_vec
        else:
            return (0,0)
                            
    #ステップが進んだ時の処理,終了判定などして、報酬,配達完了判定,行動判定を返す
    def step(self,agent_id,action):
        #配送できたかを管理する
        delivered = False
        #報酬
        reward = 0.0
        #エージェントの現在情報(座標,荷物)取得
        ay,ax = self.agents_pos[agent_id]
        cargo = self.agents_cargo[agent_id]
        #エージェントの行動集合
        day = [-1,1,0,0,0]
        dax = [0,0,-1,1,0]
        #移動候補
        nay,nax = (ay+day[action], ax+dax[action])
        #移動可否判定
        movable = True
        #境界
        if not (0 <= nay < len(self.layout)and 0 <= nax < len(self.layout[0])):
            movable = False
        #壁
        elif self.layout[nay][nax] != 1:
            movable = False
        else:
            #他エージェントが移動先にいるか
            for other_id,pos in enumerate(self.agents_pos):
                if other_id == agent_id:
                    continue
                else:
                    if pos == (nay,nax):
                        movable = False
                        break

        #移動できない場合とどまる
        if not movable:
            nay,nax = ay,ax
        #新たなエージェント座標
        self.agents_pos[agent_id] = (nay,nax)

        #----------------------------
        #新たな座標をキューに追加
        self.agents_posqueue[agent_id].append(self.agents_pos[agent_id])

        dx = [0,0,-1,1]
        dy = [-1,1,0,0]
        for k in range(4):
            ny, nx = nay + dy[k], nax + dx[k]
            #境界チェック
            if not (0 <= ny < len(self.layout) and 0 <= nx < len(self.layout[0])):
                continue
            #搬入口の上下左右1マスの距離、かつ荷物未所持
            if self.layout[ny][nx] == 3 and not cargo:
                self.agents_cargo[agent_id] = True
                break
            #搬出口にいて、かつ荷物所持
            elif self.layout[ny][nx] == 4 and cargo:
                self.agents_cargo[agent_id] = False
                reward = 1.0
                delivered = True
                self.agents_pos[agent_id] = self.random_spawn(agent_id)
                #搬出したのち、ベクトルをリセットし初期座標をキューに追加
                self.agents_posqueue[agent_id].clear()
                self.agents_posqueue[agent_id].append(self.agents_pos[agent_id])
                break
        return reward,delivered,movable
