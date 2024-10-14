import numpy as np
from sklearn.model_selection import KFold
RANDOM_STATE = 42

class KFoldSplit:
    def __init__(self, X: np.array, Y: np.array, n_splits: int):
        self.X = X
        self.Y = Y
        self.random_state= RANDOM_STATE
        self.shuffle = True
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def k_splits(self):
        X_train, X_val = [], []
        Y_train, Y_val = [], []
        train_idxs, val_idxs = [], []
        for i, (train_idx, val_idx) in enumerate(self.kf.split(self.X, self.Y)):
            X_train.append(self.X[train_idx]), X_val.append(self.X[val_idx])
            Y_train.append(self.Y[train_idx]), Y_val.append(self.Y[val_idx])
            train_idxs.append(train_idx), val_idxs.append(val_idx)
        self.info_dict = {'X Train': X_train, 'X Val': X_val, 
                          'Y Train': Y_train, 'Y Val': Y_val,
                          'Train Index': train_idxs, 'Val Index': val_idxs}
        return self.info_dict