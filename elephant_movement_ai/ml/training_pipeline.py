from sklearn.model_selection import train_test_split
from .feature_engineering import FeatureEngineering
from .risk_model import RiskModel

class TrainingPipeline:
    """Top-level ML orchestrator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.datasets_dir = config['paths'].get('datasets_dir', 'output/datasets/')
        
        self.fe = FeatureEngineering()
        self.model = RiskModel()
        
    def run(self):
        print("\n=== EXECUTING MACHINE LEARNING PIPELINE ===")
        
        X, y = self.fe.build_features(self.datasets_dir)
        
        if X.empty:
            print("No features extracted. Skipping ML Pipeline.")
            return None
            
        print(f"Dataset shape: {X.shape}, Conflicts: {int(y.sum())}/{len(y)}")
        
        # Chronological split is better for timeseries, but random is okay for PoC validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.train(X_train, y_train)
        self.model.evaluate(X_test, y_test)
        
        return self.model
