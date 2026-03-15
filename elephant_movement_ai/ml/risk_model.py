from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

class RiskModel:
    """Binary classifier to predict conflict outbreaks 24hrs in advance."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if X_train.empty or len(y_train) == 0:
            print("Warning: Empty training set. Skipping ML Train.")
            return
            
        print("Training Random Forest Risk Model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        if not self.is_trained or X_test.empty:
            return
            
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1] if len(self.model.classes_) == 2 else preds
        
        print("\n--- ML Model Evaluation ---")
        try:
            print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")
            print(classification_report(y_test, preds))
        except ValueError as e:
            print(f"Could not calculate metrics (likely only 1 class present in test set): {e}")
            
    def predict_risk(self, current_features: pd.DataFrame) -> np.ndarray:
        """Returns risk probabilities for the passed environment array."""
        if not self.is_trained:
            return np.zeros(len(current_features))
        
        if len(self.model.classes_) == 1:
            return np.zeros(len(current_features)) if self.model.classes_[0] == 0 else np.ones(len(current_features))
            
        return self.model.predict_proba(current_features)[:, 1]
