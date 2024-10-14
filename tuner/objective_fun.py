import numpy as np
import metric
import crossval

class ObjectiveFunction:
    def __init__(self, X: np.array, Y: np.array, n_splits: int, shuffle: bool, study: str, model, metrics: list, weights: list):
        self.X = X
        self.Y = Y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.study = study
        self.model = model
        self.model_type = str.lower(str(type(self.model)))
        assert len(metrics) == len(weights), 'shape mismatch'
        self.metrics = metrics
        self.weights = weights
        if 'regressor' in self.model_type:
            self.model_type = 'regressor'
            self.metrics = list(metric.regressor.DICT_METRICS.keys())
            for metric in self.metrics:
                if metric not in regressor.DICT_METRICS:
                    raise ValueError('Unsupported Metric Type')
        elif 'classifier' in self.model_type:
            self.model_type = 'classifier'
            self.metrics = list(classifier.DICT_METRICS.keys())
            for metric in self.metrics:
                if metric not in metric.classifier.DICT_METRICS:
                    raise ValueError('Unsupported Metric Type')
        else:
            raise ValueError('Unsupported Model Type')
        self.train_score = []
        self.val_score = []

    def objective(self, trial):
        dict_metric = {}
        if model_type == 'regressor':
            list_metric = list(regressor.DICT_METRICS.keys())
            splitter = kfold.KFoldSplit(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            k_folds = splitter.k_splits()
            self.X_train, self.X_val = k_folds['X Train'], k_folds['X Val']
            self.Y_train, self.Y_val = k_folds['Y Train'], k_folds['Y Val']
            
            
                
                
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        elif model_type == 'classifier':
            list_metric = list(classifier.DICT_METRICS.keys())
            splitter = stratifiedkfold.StratifiedKFoldSplit(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            k_folds = splitter.k_splits()
            X_train, X_val = k_folds['X Train'], k_folds['X Val']
            Y_train, Y_val = k_folds['Y Train'], k_folds['Y Val']
        else:
            raise ValueError('Unsupported Model Type')
        
        
        for i in range(len(X_train)):
            model.fit(X_train, Y_train)
            Y_pred_train = model.predict(X_train)
            Y_pred_val = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                Y_pred_proba_train = model.predict_proba(X_train)
                Y_pred_proba_val = model.predict_proba(X_val)
            
            
        
        
            
    
    