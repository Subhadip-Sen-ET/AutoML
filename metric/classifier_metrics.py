import numpy as np
import sklearn.metrics as metrics
DICT_METRICS = {'precision': ['binary', 'micro', 'macro', 'weighted', 'samples', None],
                'recall': ['binary', 'micro', 'macro', 'weighted', 'samples', None],
                'f1': ['binary', 'micro', 'macro', 'weighted', 'samples', None],
                'accuracy': [],
                'balanced accuracy': [], 
                'roc auc': ['binary', 'micro', 'macro', 'weighted', 'samples', None]}

class Metrics(metrics):
    def __init__(self, Y_true: np.array, Y_pred: np.array, 
                 Y_pred_proba: np.array, sample_weight=None):
        super().__init__()
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        self.Y_pred_proba = Y_pred_proba
        assert Y_true.shape == Y_pred.shape, 'Shape Mismatch'
        self.sample_weight = None if sample_weight == None else sample_weight
        
    def precision(self, strategy=None):
        if strategy in DICT_METRICS['precision']:
            self.strategy = strategy
        else:
            raise ValueError('Unsupported Strategy')
        precision = self.precision_score(Y_true=self.Y_true, 
                                         Y_pred=self.Y_pred, 
                                         sample_weight=self.sample_weight, 
                                         strategy=self.strategy)
        return precision
    
    def recall(self, strategy=None):
        if strategy in DICT_METRICS['recall']:
            self.strategy = strategy
        else:
            raise ValueError('Unsupported Strategy')
        recall = self.recall_score(Y_true=self.Y_true, 
                                   Y_pred=self.Y_pred, 
                                   sample_weight=self.sample_weight, 
                                   strategy=self.strategy)
        return recall
    
    def f1(self, strategy=None):
        if strategy in DICT_METRICS['f1']:
            self.strategy = strategy
        else:
            raise ValueError('Unsupported Strategy')
        recall = self.f1_score(Y_true=self.Y_true, 
                               Y_pred=self.Y_pred, 
                               sample_weight=self.sample_weight, 
                               strategy=self.strategy)
        return recall
    
    def accuracy(self):
        accuracy = self.accuracy_score(Y_true=self.Y_true, 
                                       Y_pred=self.Y_pred, 
                                       sample_weight=self.sample_weight, 
                                       strategy=self.strategy)
        return accuracy
    
    def balanced_accuracy(self):
        balanced_accuracy = self.balanced_accuracy_score(Y_true=self.Y_true, 
                                                         Y_pred=self.Y_pred, 
                                                         sample_weight=self.sample_weight, 
                                                         strategy=self.strategy)
        return balanced_accuracy
    
    def roc_auc(self, average=None):
        roc_auc = self.roc_auc_score(Y_true=self.Y_true, 
                                     Y_pred=self.Y_pred_proba,
                                     sample_weight=self.sample_weight,
                                     average=self.average)
        return roc_auc
        
        


