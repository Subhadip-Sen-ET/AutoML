Class HyperparameterTuner:
    def __init__(self, n_trials: int, n_splits: int, study: str):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.study = study
        self.best_training_score = None
        self.best_validation_score = None

    def tune(self, X, Y, classifier):
        def objective(trial):
            
            clf = classifier(trial)
            clf = clf.classifier(trial)
            skf = StratifiedKFold(n_splits=self.n_splits)
            
            list_train_score   = []
            list_val_score     = []
            
            for train_idx, test_idx in skf.split(X, Y):

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

                clf.fit(X_train, Y_train)
                
                Y_train_pred = clf.predict(X_train)
                Y_test_pred  = clf.predict(X_test)
                
                train_score  = roc_auc(Y_train, Y_train_pred)
                val_score    = roc_auc(Y_test, Y_test_pred)
                
                list_train_score.append(train_score)
                list_val_score.append(val_score)
                
            mean_training_score = np.mean(list_train_score)
            mean_validation_score = np.mean(list_val_score)
                
            if self.best_validation_score is None or mean_validation_score > self.best_validation_score:
                self.best_training_score = mean_training_score
                self.best_validation_score = mean_validation_score
                
            # Return the average accuracy of both classifiers
            return mean_validation_score

        self.study = optuna.create_study(direction='maximize')
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

