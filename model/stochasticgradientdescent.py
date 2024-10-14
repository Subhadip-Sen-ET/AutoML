from sklearn.linear_model import SGDClassifier, SGDRegressor

class StochasticGradientDescent:
    
    def __init__(self, trial):  
        self.penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None]) 
        self.alpha = trial.suggest_float('alpha', 1e-4, 1) 
        if self.penalty == 'elasticnet':
            self.l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0) 
        self.fit_intercept = trial.suggest_categorical('fit_intercept', [True, False]) 
        self.max_iter = trial.suggest_int('max_iter', 1e3, 1e6) 
        self.tol = trial.suggest_float('tol', 1e-3, 1e-6) 
        self.shuffle = trial.suggest_categorical('shuffle', [True, False])   
        self.n_jobs = -1, 
        self.random_state = 42, 
        self.learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 
                                                                    'adaptive']) 
        if self.learning_rate in ['constant', 'invscaling', 'adaptive']:
            self.eta0 = trial.suggest_float('eta0', 0.0, 1e3) 
        if self.learning_rate == 'invscaling':
            self.power_t = trial.suggest_float('power_t', -1e3, 1e3) 
        self.early_stopping = trial.suggest_categorical('early_stopping', [True, False]) 
        if self.early_stopping == True:
            self.validation_fraction = trial.suggest_float('validation_fraction', 1e-3, 999e-3) 
        if self.early_stopping == True:
            self.n_iter_no_change = trial.suggest_int('n_iter_no_change', 1, 1e2)  
        self.warm_start = True 
        
        self.params = {
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'shuffle': self.shuffle,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'power_t': self.power_t,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_inter_no_change,
            'warm_start': self.warm_start            
        }
        
    def classifier(self, trial):
        self.params['loss'] = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber', 
                                                                 'squared_hinge', 'perceptron', 'squared_error', 
                                                                 'huber', 'epsilon_insensitive', 
                                                                 'squared_epsilon_insensitive'])
        if self.params['loss'] in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
            self.params['epsilon'] = trial.suggest_float('epsilon', 0.0, 1e3)
        self.params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])  
        self.classifier = SGDClassifier(**self.params)
        
        return self.classifier
    
    def regressor(self, trial):
        self.params['loss'] = trial.suggest_categorical('loss', ['squared_error', 'huber', 
                                                                 'epsilon_insensitive',
                                                                 'squared_epsilon_insensitive'])
        if self.params['loss'] in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
            self.params['epsilon'] = trial.suggest_float('epsilon', 0.0, 1e3)
        self.regressor = SGDRegressor(**self.params)
        
        return self.regressor