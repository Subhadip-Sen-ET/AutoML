from sklearn.model_selection import StratifiedKFold


class KFoldCrossVal:
    
    def __init__(self, n_splits: int, stratify: None):
        
        self.n_splits = n_splits
        self.stratify = stratify
        
    def create_crossval_sets(self):
        
        if self.stratify == None:
            
        else:
            
        
        