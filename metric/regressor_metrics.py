import numpy as np
import sklearn.metrics as metrics
DICT_METRICS = {'mean_squared_error': ['raw_values', 'uniform_average', None], 
                'mean_absolute_percentage_error': ['raw_values', 'uniform_average', None], 
                'mean_absolute_error': ['raw_values', 'uniform_average', None]}

class Metrics(metrics):
    def __init__(self, Y_true: np.array, Y_pred: np.array, sample_weight=None):
        super().__init__()
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        self.sample_weight = None if sample_weight == None else sample_weight
        
    def mean_squared_error(self, multioutput='uniform_average'):
        if multioutput in DICT_METRICS['mean_squared_error']:
            self.multioutput = multioutput
        else:
            raise ValueError('Unsupported Multioutput Type')
        mean_squared_error = self.mean_squared_error(Y_true=self.Y_true, 
                                                     Y_pred=self.Y_pred, 
                                                     sample_weight=self.sample_weight, 
                                                     multioutput=self.multioutput)
        return mean_squared_error
    
    def mean_absolute_percentage_error(self, multioutput='uniform average'):
        if multioutput in DICT_METRICS['mean_absolute_percentage_error']:
            self.multioutput = multioutput
        else:
            raise ValueError('Unsupported Multioutput Type')
        mean_absolute_percentage_error = self.mean_absolute_percentage_error(Y_true=self.Y_true,
                                                                             Y_pred=self.Y_pred,
                                                                             sample_weight=self.sample_weight,
                                                                             multioutput=self.multioutput)
        return mean_absolute_percentage_error
    
    def mean_absolute_error(self, multioutput='uniform_average'):
        if multioutput in DICT_METRICS['mean_absolute_error']:
            self.multioutput = multioutput
        else:
            raise ValueError('Unsupported Multioutput Type')
        mean_absolute_error = self.mean_absolute_error(Y_true=self.Y_true,
                                                       Y_pred=self.Y_pred,
                                                       sample_weight=self.sample_weight,
                                                       multioutput=self.multioutput)
        return mean_absolute_error
            




