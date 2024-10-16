from metric import *
from crossval import *

class ObjectiveFunction:
    def __init__(self, X: np.array, Y: np.array, model, n_splits: int=5, 
                 shuffle: bool=True, metrics: list=['accuracy'], weights: list=[1]):
        self.X = X
        self.Y = Y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.study = study
        self.model = model
        self.model_type = str.lower(str(type(self.model)))
        assert sum(weights) == 1, 'Sum of Weights should be 1'
        assert len(metrics) == len(weights), 'Shape Mismatch'
        self.metrics = metrics
        self.weights = weights
        self.train_score = []
        self.val_score = []
        self.Y_train_pred = []
        self.Y_val_pred = []
        self.train_error = 0
        self.val_error = 0
            
        def regressor_score(self):
            reg_metrics = list(regressor_metrics.DICT_METRICS.keys())
            
            if 'regressor' in self.model_type:
                self.model_type = 'regressor'
                for metric in range(len(self.metrics)):
                    if metric in reg_metrics:
                        splitter = kfold.KFoldSplit(X=self.X,
                                                    Y=self.Y,
                                                    n_splits=self.n_splits)
                        k_folds = splitter.k_splits()
                        self.X_train, self.X_val = k_folds['X Train'], k_folds['X Val']
                        self.Y_train, self.Y_val = k_folds['Y Train'], k_folds['Y Val']
                        for i in range(len(self.X_train)):
                            self.model.fit(self.X_train[i], self.Y_train[i])
                            Y_pred_train = self.model.predict(self.X_train[i])
                            Y_pred_val = self.model.predict(self.X_val[i])
                            self.Y_train_pred.append(Y_pred_train)
                            self.Y_val_pred.append(Y_pred_val)
                            create_metrics_reg_train = regressor_metrics.Metrics(Y_true=Y_train[i], 
                                                                                 Y_pred=Y_pred_train)
                            create_metrics_reg_val = regressor_metrics.Metrics(Y_true=Y_val[i],
                                                                               Y_pred=Y_val_pred)
                            for i in range(len(self.metrics)):
                                if self.metrics[i] == regressor_metrics.DICT_METRICS[0]:
                                    train_error = create_metrics_reg_train.mean_squared_error()
                                    val_error = create_metrics_reg_val.mean_squared_error()
                                elif self.metrics[i] == regressor_metrics.DICT_METRICS[1]:
                                    train_error = create_metrics_reg_train.mean_absolute_percentage_error()
                                    val_error = create_metrics_reg_val.mean_absolute_percentage_error()
                                elif self.metrics[i] == regressor_metrics.DICT_METRICS[2]:
                                    train_error = create_metrics_reg_train.mean_absolute_error()
                                    val_error = create_metrics_reg_val.mean_absolute_error()
                                else: 
                                    raise ValueError('Unsupported Loss Function')
                                self.train_error += self.train_error * self.weights[i]
                                self.val_error += self.val_error * self.weights[i]
                            self.train_score.append(self.train_error)
                            self.val_score.append(self.val_error)
                    else:
                        raise ValueError('Unsupported Metric Type')
                self.train_error += self.train_error * self.weights[i]
                self.val_error += self.val_error * self.weights[i]
            else:
                raise ValueError('Unsupported Model Type')
            
            self.mean_train_score = mean(self.train_error)
            self.mean_val_score = mean(self.val_error)
            
            return (self.mean_train_score, self.mean_val_score)
        
        def classifier_score(self):
            clf_metrics = list(classifier_metrics.DICT_METRICS.keys())
            
            if 'classifier' in self.model_type:
                self.model_type = 'classifier'
                unq_cat = np.unique(self.Y)
                if len(unq_cat) == 2:
                    strategy = None
                elif len(unq_cat) > 2:
                    strategy = 'weighted'
                for metric in range(len(self.metrics)):
                    if metric in clf_metrics:
                        splitter = stratified_kfold.StratifiedKFoldSplit(X=self.X, 
                                                                         Y=self.Y, 
                                                                         n_splits=self.n_splits)
                        kfold = splitter.k_splits()
                        self.X_train, self.X_val = k_folds['X Train'], k_folds['X Val']
                        self.Y_train, self.Y_val = k_folds['Y Train'], k_folds['Y Val']
                        for i in range(len(self.X_train)):
                            self.model.fit(self.X_train[i], self.Y_train[i])
                            Y_pred_train = self.model.predict(self.X_train[i])
                            Y_pred_val = self.model.predict(self.X_val[i])
                            if hasattr(self.model, 'predict_proba'):
                                self.Y_train_proba_pred = []
                                self.Y_val_proba_pred = []
                                Y_pred_proba_train = self.model.predict_proba(X_train[i])
                                Y_pred_proba_val = self.model.predict_proba(X_val[i]) 
                            self.Y_train_pred.append(Y_pred_train)
                            self.Y_val_pred.append(Y_pred_val)
                            self.Y_train_proba_pred.append(Y_pred_proba_train)
                            self.Y_val_proba_pred.append(Y_pred_proba_val)
                            create_metrics_clf_train = classifier_metrics.Metrics(Y_true=Y_train[i], 
                                                                                 Y_pred=Y_pred_train,
                                                                                 Y_pred_proba=self.Y_train_proba_pred[i])
                            create_metrics_clf_val = classifier_metrics.Metrics(Y_true=Y_val[i],
                                                                               Y_pred=Y_val_pred,
                                                                               Y_pred_proba=self.Y_pred_proba_val[i])
                            for i in range(len(self.metrics)):
                                if self.metrics[i] == classifier_metrics.DICT_METRICS[0]:
                                    train_error = create_metrics_clf_train.precision(strategy=strategy)
                                    val_error = create_metrics_clf_val.precision(strategy=strategy)
                                elif self.metrics[i] == classifier_metrics.DICT_METRICS[1]:
                                    train_error = create_metrics_clf_train.recall(strategy=strategy)
                                    val_error = create_metrics_clf_val.recall(strategy=strategy)
                                elif self.metrics[i] == classifier_metrics.DICT_METRICS[2]:
                                    train_error = create_metrics_clf_train.f1(strategy=strategy)
                                    val_error = create_metrics_clf_val.f1(strategy=strategy)
                                elif self.metrics[i] == classifier_metrics.DICT_METRICS[3]:
                                    train_error = create_metrics_clf_train.accuracy()
                                    val_error = create_metrics_clf_val.accuracy()
                                elif self.metrics[i] == classifier_metrics.DICT_METRICS[4]:
                                    train_error = create_metrics_clf_train.balanced_accuracy()
                                    val_error = create_metrics_clf_val.balanced_accuracy()
                                elif self.metrics[i] == classifier_metrics.DICT_METRICS[5]:
                                    train_error = create_metrics_clf_train.roc_auc()
                                    val_error = create_metrics_clf_val.roc_auc()
                                else: 
                                    raise ValueError('Unsupported Loss Function')
                            self.train_error += self.train_error * self.weights[i]
                            self.val_error += self.val_error * self.weights[i]
                            
            else:
                raise ValueError('Unsupported Model Type')
            
            self.mean_train_score = mean(self.train_error)
            self.mean_val_score = mean(self.val_error)
            
            return (self.mean_train_score, self.mean_val_score)
        
        
            
        
        
            
    
    