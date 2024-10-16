from tuner import objective_fun
import optuna
import numpy as np

class HyperparameterTuner:
    def __init__(self, n_trials: int=100, study: str='Unknown Study',
                 direction: str='maximize'):
        self.n_trials = n_trials
        self.study = study
        self.direction = direction
        self.best_training_score = None
        self.best_validation_score = None

    def tune(self, X: np.array, Y:np.array, model):
        self.X = X
        self.Y = Y
        self.model = model
        objective = objective_fun.ObjectiveFunction(X=self.X, 
                                                    Y=self.Y, 
                                                    model=self.model)
        if 'classifier' in objective.model_type:
            mean_training_score = objective.classifier_score()[0]
            mean_validation_score = objective.classifier_score()[1]
        elif 'regressor' in objective.model_type:
            mean_training_score = objective.regressor_score()[0]
            mean_validation_score = objective.regressor_score()[1]
        else:
            raise ValueError('Invalid Model Type')
        
        if self.best_validation_score is None or mean_validation_score > self.best_validation_score:
            self.best_training_score = mean_training_score
            self.best_validation_score = mean_validation_score

        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)

    def get_best_params(self):
        if self.study is None:
            raise ValueError("Tuning has not been run yet.")
        return self.study.best_params
    
    def get_best_score(self):
        if self.study is None:
            raise ValueError("Tuning has not been run yet.")
        return self.study.best_value
    
    def get_best_training_score(self):
        if self.best_training_score is None:
            raise ValueError("Tuning has not been run yet.")
        return self.best_training_score
    
    def get_best_validation_score(self):
        if self.best_validation_score is None:
            raise ValueError("Tuning has not been run yet.")
        return self.best_validation_score

