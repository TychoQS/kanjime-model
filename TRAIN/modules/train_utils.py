class EarlyStopping:
    """Early Stopping implementation for training"""
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        
    def __call__(self, val_accuracy):
        score = val_accuracy
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"  [Early Stopping] Baseline set: {score:.4f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [Early Stopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  [Early Stopping] TRIGGERED! Best score: {self.best_score:.4f}")
                return True
        else:
            if self.verbose:
                print(f"  [Early Stopping] Improvement! {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.counter = 0
        return False