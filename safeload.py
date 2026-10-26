import pickle, io, gzip
class SafeUnpickler(pickle.Unpickler):
    """numpy配列と組み込み型だけ復元を許可（任意コード実行を防ぐ）"""
    def find_class(self, module, name):
        if module.split('.')[0] == 'numpy' or (module == 'builtins' and name in {'tuple','list','dict','set','frozenset','int','float','bool','str'}):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'許可されていないクラス: {module}.{name}')
def safe_load(path):
    opener = gzip.open if str(path).endswith('.gz') else open   # .pkl.gz も .pkl も読める
    with opener(path, 'rb') as f:
        return SafeUnpickler(f).load()
