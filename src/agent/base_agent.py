import numpy as np

class BaseAgent:
    def __init__(self):
        #Qテーブル(辞書型)
        self.q_table = {}
        #エピソードバッファ((観測状態,行動)の配列)
        self.episode_buffer = []
        #周囲8マスの最新情報
        self.obs = None
        #荷物を持っているか
        self.cargo = False

        #エージェントの行動集合数
        self.action_space_num = 5 

    #エージェントのバッファをリセット
    def reset_history(self):
        self.episode_buffer = []
        self.obs = None
        self.cargo = False

    #環境から得た観測情報を状態に変換し、観測情報+荷物+相対ベクトルに変換する関数
    # 最終的にタプルに入る順番
    #(北西, 北, 北東, 西, 東, 南西, 南, 南東)
    '''
    観測情報obsは以下である。
    [[北西,北,北東],
     [西, 自分, 東],
     [南西,南,南東]]
    '''
    def convert_state(self, obs, cargo, rel_vec=None):
        state_element = []
        for i,dy in enumerate([-1,0,1]):
            for j,dx in enumerate([-1,0,1]):
                if dy == 0 and dx == 0:
                    continue
                state_element.append(obs[i][j])
        return (tuple(state_element), cargo, rel_vec)
    #Qテーブルに状態空間が存在しない場合作成
    def initialize_state(self,state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_num)
        
    #q_tableから(状態,荷物)をひっくるめた状態の部分を参照し行動のQ値を得る
    #Qテーブルは、(上,下,左,右,停止)である
    #その後softmax_policyにわたして最適な行動を返す
    def get_action(self, temperature, rel_vec=None):
        #(状態,荷物)の状態キー
        state_key = self.convert_state(self.obs, self.cargo, rel_vec)
        #Qテーブルに状態キーに対する状態空間が存在しない場合作成
        self.initialize_state(state_key)
        #Qテーブルから状態キーに対する状態空間参照(行動のQ値)
        q_values = self.q_table[state_key].copy()
        #観測情報から行動不可になるQ値を除外する
        #上行動可能か
        if self.obs[0][1] != 1:
            q_values[0] = -np.inf
        #下行動可能か
        if self.obs[2][1] != 1:
            q_values[1] = -np.inf
        #左行動可能か
        if self.obs[1][0] != 1:
            q_values[2] = -np.inf
        #右行動可能か
        if self.obs[1][2] != 1:
            q_values[3] = -np.inf
        #停止行動は確実に可能なため省略
        
        #softmax分布により最適行動求める
        action = self.softmax_policy(q_values,temperature)

        return action
            
    #softmax分布(Qテーブルの一部から最適な行動actionを返す)
    def softmax_policy(self, q_values, temperature):
        #Q値の指数乗の合計
        sum_q = 0;
        #各行動の確率(softmax分布)
        #全て行動不能の場合、ランダムな行動を返す
        if np.all(q_values == -np.inf):
            probability = np.ones(self.action_space_num) / 5
            return np.random.choice(self.action_space_num, p=probability)

        valid_q = q_values[q_values != -np.inf]
        max_q = np.max(valid_q)

        exp_q = np.zeros(self.action_space_num)

        for idx in range(self.action_space_num):
            exp_q[idx] = np.exp((q_values[idx] - max_q) / temperature)
        #指数関数を確率に変換
        probability = exp_q / np.sum(exp_q)
        #行動を選択
        action = np.random.choice(self.action_space_num,p=probability)
        
        return action 
