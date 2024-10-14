from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class GradientBoosting:
    
    def __init__(self, trial):
        self.learning_rate = trial.suggest_float('learning_rate', 0, 1e3)
        self.n_estimators = trial.suggest_int('n_estimators', 1, 1e3)
        self.subsample = trial.suggest_float('subsample', 0.0, 1.0) 
        self.criterion = trial.suggest__categorical('criterion', ['friedman_mse', 'squared_error'])
        self.min_samples_split = trial.suggest_int('min_samples_split', 2, 1e2)
        self.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 1e2)
        self.min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        self.max_depth = trial.suggest_int('max_depth', 1, 30)
        self.random_state = 42
        self.max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        self.verbose = 0,
        self.warm_start = True
        self.n_iter_no_change = trial.suggest_int('n_iter_no_change', 1, 1e2)  
        if self.n_iter_no_change != None:
            self.validation_fraction = trial.suggest_float('validation_fraction', 1e-3, 999e-3)
        self.tol = trial.suggest_float('tol', 1e-3, 1e-6)
        self.ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)
        
        self.params = {
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'max_features': self.max_features,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'n_iter_no_change': self.n_iter_no_change,
            'validation_fraction': self.validation_fraction,
            'tol': self.tol,
            'ccp_alpha': self.ccp_alpha
        }
        
    def classifier(self, trial):
        self.params['loss'] = trial.suggest_categorical('loss', ['log_loss', 'exponential'])  
        self.classifier = GradientBoostingClassifier(**self.params)
        
        return self.classifier
        
    def regressor(self, trial):
        self.params['loss'] = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
        if self.params['loss'] in ['huber', 'quantile']:
            self.params['alpha'] = trial.suggest_float('alpha', 0.0, 1.0)
        self.regressor = GradientBoostingRegressor(**self.params)
        
        return self.regressor
