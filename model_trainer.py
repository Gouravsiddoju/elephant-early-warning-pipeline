import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def prepare_train_test(df: pd.DataFrame):
    """
    Chronological split at 2016-01-01.
    Encode grid_id labels with LabelEncoder.
    Handle class imbalance: only keep grid cells with >= 10 transitions.
    Scale features with StandardScaler (fit on train only).
    """
    print(f"[{datetime.now().isoformat()}] Preparing train/test split (Chronological)...")
    
    # 1. Filter out rare classes
    class_counts = df['to_grid'].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    
    initial_len = len(df)
    df = df[df['to_grid'].isin(valid_classes)].copy()
    print(f"[{datetime.now().isoformat()}] Dropped {initial_len - len(df)} rows belonging to classes with <10 samples.")
    
    # 2. Encode Labels
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['to_grid'])
    
    # Save encoder
    joblib.dump(le, 'label_encoder.pkl')
    
    # 3. Chronological Split — use 80th percentile of sorted dates (works regardless of year range)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], utc=True, errors='coerce')
    sorted_dates = df['Date_Time'].sort_values().reset_index(drop=True)
    split_idx = int(len(sorted_dates) * 0.80)
    split_date = sorted_dates.iloc[split_idx]
    print(f"[{datetime.now().isoformat()}] Chronological split date (80/20): {split_date}")
        
    train_mask = df['Date_Time'] < split_date
    test_mask = df['Date_Time'] >= split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"[{datetime.now().isoformat()}] Train set: {len(train_df)} rows. Test set: {len(test_df)} rows.")
    
    # 4. Feature Selection
    exclude_cols = ['elephant_id', 'Date_Time', 'from_grid', 'to_grid', 'target', 'grid_centroid_lon', 'grid_centroid_lat']
    features = [c for c in df.columns if c not in exclude_cols]
    
    for col in features:
        if train_df[col].dtype == 'object':
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    X_train = train_df[features].values
    y_train = train_df['target'].values
    
    X_test = test_df[features].values
    y_test = test_df['target'].values
    
    # 5. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'feature_names.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier
    """
    print(f"[{datetime.now().isoformat()}] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    print(f"[{datetime.now().isoformat()}] Training complete.")
    return rf

def train_gradient_boost(X_train, y_train):
    """
    Train a Gradient Boosting Classifier
    """
    print(f"[{datetime.now().isoformat()}] Training Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    print(f"[{datetime.now().isoformat()}] Training complete.")
    return gb

def evaluate_model(model, X_test, y_test, le_path='label_encoder.pkl', feat_path='feature_names.pkl'):
    """
    Compute and print accuracy metrics, per class F1 for top 20, 
    plot confusion matrix and feature importance.
    """
    print(f"[{datetime.now().isoformat()}] Evaluating Model...")
    
    try:
        le = joblib.load(le_path)
        features = joblib.load(feat_path)
    except:
        print("Could not load label encoder or feature names for evaluation.")
        le = None
        features = [f"Feature_{i}" for i in range(X_test.shape[1])]
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    top1 = accuracy_score(y_test, y_pred)
    sorted_probs = np.argsort(y_prob, axis=1)
    
    top3_correct = [y_test[i] in sorted_probs[i, -3:] for i in range(len(y_test))]
    top3 = np.mean(top3_correct)
    
    top5_correct = [y_test[i] in sorted_probs[i, -5:] for i in range(len(y_test))]
    top5 = np.mean(top5_correct)
    
    metrics = {
        'Top-1 Accuracy': top1,
        'Top-3 Accuracy': top3,
        'Top-5 Accuracy': top5
    }
    
    print("\n--- Model Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\n--- Per-Class Metrics (Top 20 most frequent) ---")
    unique, counts = np.unique(y_test, return_counts=True)
    top20_classes = unique[np.argsort(counts)[-20:]]
    top20_labels = le.inverse_transform(top20_classes) if le else top20_classes
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for cls, lbl in zip(top20_classes, top20_labels):
        cls_str = str(cls)
        if cls_str in report_dict:
            f1 = report_dict[cls_str]['f1-score']
            print(f"Class: {lbl} | F1-Score: {f1:.4f} | Support: {report_dict[cls_str]['support']}")
            
    print(f"[{datetime.now().isoformat()}] Generating Evaluation Artifacts...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        top_feats = [features[i] for i in indices]
        top_imps = importances[indices]
        
        axes[0].barh(range(len(indices)), top_imps[::-1], align='center')
        axes[0].set_yticks(range(len(indices)))
        axes[0].set_yticklabels(top_feats[::-1])
        axes[0].set_title("Top 20 Feature Importances")
        axes[0].set_xlabel("Relative Importance")
        
    mask = np.isin(y_test, top20_classes)
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    
    if len(y_test_sub) > 0:
        cm = confusion_matrix(y_test_sub, y_pred_sub, labels=top20_classes)
        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[1], 
                    xticklabels=top20_labels, yticklabels=top20_labels)
        axes[1].set_title("Confusion Matrix (Top 20 Classes)")
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        axes[1].tick_params(axis='x', rotation=90)
        
    plt.tight_layout()
    plt.savefig('evaluation_report.png')
    print(f"[{datetime.now().isoformat()}] Saved evaluation_report.png")
    
    joblib.dump(model, 'elephant_model.pkl')
    print(f"[{datetime.now().isoformat()}] Saved model to elephant_model.pkl")
    
    return metrics

if __name__ == "__main__":
    pass
