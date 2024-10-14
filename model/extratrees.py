from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

class ExtraTrees:
    
    def __init__(self, trial):
        self.n_estimators = trial.suggest_int('n_estimators', 1, 1e3)
        self.max_depth = trial.suggest_int('max_depth', 1, 30)
        self.min_samples_split = trial.suggest_int('min_samples_split', 2, 1e2)
        self.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 1e2)
        self.min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        self.max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        self.bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        if self.bootstrap == True:
            self.oob_score = trial.suggest_categorical('oob_score', [True, False])
        else:
            self.oob_score = False
        self.n_jobs = -1
        self.random_state = 42
        self.verbose = 0
        self.warm_start = True
        self.ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)  
        self.params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'ccp_alpha': self.ccp_alpha
        }
        
        
    def classifier(self, trial):      
        self.params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        self.params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        self.classifier = ExtraTreesClassifier(**self.params) 
        
        return self.classifier
    
    def regressor(self, trial): 
        self.params['criterion'] = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
        self.regressor = ExtraTreesRegressor(**self.params)
        
        return self.regressor