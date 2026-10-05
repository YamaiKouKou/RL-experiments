import abc
"""
学習アルゴリズムの抽象クラス定義

    Methods:
     - train_episode(env,agents,temperature) ->
     - update(...) ->
     - compute_gradient(loss) ->
"""
class BaseAlgorithm(abc.ABC):
    def __init__(self):
        self.optimizer = None
        self.criterion = None
    
    @abc.abstractmethod
    def run(self,env,agents,temperature,file_path):
        """
        1エピソード分の実行と学習を統括する抽象メソッド
        子クラス(QPSPAlgorithmや将来のDQN)で必ずオーバーライドさせる
        """
        pass

    @abc.abstractmethod
    def update(self,*args,**kwargs):
        """
        パラメータ(QテーブルやNNの重み)を更新する抽象メソッド
        """
        pass

    def compute_gradient(self,loss):
        """
        DQN用抽象メソッド
        """
        if self.optimizer is not None and loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
