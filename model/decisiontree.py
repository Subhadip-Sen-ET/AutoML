from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DecisionTree:
    
    def __init__(self, trial):    
        self.splitter = trial.suggest_categorical('splitter', ['best', 'random'])
        self.max_depth = trial.suggest_int('max_depth', 1, 30) 
        self.min_samples_split = trial.suggest_int('min_samples_split', 2, 1e2) 
        self.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 1e2)
        self.min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5) 
        self.max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        self.random_state = 42
        self.ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)
        self.params = {
            'splitter': self.splitter,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'random_state': self.random_state,
            'ccp_alpha': self.ccp_alpha
        }
        
    def classifier(self, trial):
        self.params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        self.params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])
        self.classifier = DecisionTreeClassifier(**self.params)
        
        return self.classifier
    
    def regressor(self, trial): 
        self.params['criterion'] = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        self.regressor = DecisionTreeRegressor(**self.params)
        
        return self.regressor