import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime
import warnings

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

SEQ_LEN = 10

class ElephantLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(ElephantLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def create_sequences(df, features, seq_len=SEQ_LEN):
    """
    Groups by elephant_id and creates rolling windows of length seq_len.
    Target is the 'target' of the LAST step in the window.
    """
    X_seq = []
    y_seq = []
    
    for elephant_id, group in df.groupby('elephant_id'):
        group = group.sort_values('Date_Time').reset_index(drop=True)
        if len(group) < seq_len:
            continue
            
        feat_vals = group[features].values
        targ_vals = group['target'].values
        
        for i in range(len(group) - seq_len + 1):
            X_seq.append(feat_vals[i:i+seq_len])
            y_seq.append(targ_vals[i+seq_len-1])
            
    return np.array(X_seq), np.array(y_seq)

def prepare_train_test(df: pd.DataFrame):
    """
    Chronological split at 2016-01-01.
    Encode grid_id labels with LabelEncoder.
    Handle class imbalance: only keep grid cells with >= 10 transitions.
    Scale features with StandardScaler (fit on train only).
    """
    print(f"[{datetime.now().isoformat()}] Preparing train/test split (Chronological) for LSTM...")
    
    # 0. Drop rows where from_grid or to_grid is '0', NaN, or empty
    invalid_mask = (
        df['from_grid'].astype(str).isin(['0', 'nan', '']) |
        df['to_grid'].astype(str).isin(['0', 'nan', ''])
    )
    dropped = invalid_mask.sum()
    if dropped > 0:
        print(f"[{datetime.now().isoformat()}] WARNING: Dropping {dropped} rows with invalid grid IDs ('0' or NaN).")
    df = df[~invalid_mask].copy()
    
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
    
    print(f"[{datetime.now().isoformat()}] Train base: {len(train_df)} rows. Test base: {len(test_df)} rows.")
    
    # 4. Feature Selection
    exclude_cols = ['elephant_id', 'Date_Time', 'from_grid', 'to_grid', 'target', 'grid_centroid_lon', 'grid_centroid_lat']
    features = [c for c in df.columns if c not in exclude_cols]
    
    for col in features:
        if train_df[col].dtype == 'object':
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    # 5. Scale Features
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'feature_names.pkl')
    
    # 6. Create Sequences
    X_train_seq, y_train_seq = create_sequences(train_df, features, seq_len=SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(test_df, features, seq_len=SEQ_LEN)
    
    print(f"[{datetime.now().isoformat()}] Sequences Created - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq

def train_lstm(X_train, y_train, input_dim, output_dim, epochs=10, batch_size=256):
    """
    Train the LSTM Classifier with LR scheduler and early stopping.
    """
    print(f"[{datetime.now().isoformat()}] Training PyTorch LSTM Classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().isoformat()}] Using device: {device}")
    
    model = ElephantLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Early stopping setup
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{datetime.now().isoformat()}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # LR scheduler step
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{datetime.now().isoformat()}] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"[{datetime.now().isoformat()}] Training complete. Best loss: {best_loss:.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), 'elephant_lstm.pt')
    print(f"[{datetime.now().isoformat()}] Saved model weights to elephant_lstm.pt")
    
    return model

def evaluate_model(model, X_test, y_test, le_path='label_encoder.pkl', feat_path='feature_names.pkl'):
    """
    Compute and print accuracy metrics, per class F1 for top 20, 
    plot confusion matrix.
    """
    print(f"[{datetime.now().isoformat()}] Evaluating Model in batches to save memory...")
    
    try:
        le = joblib.load(le_path)
        features = joblib.load(feat_path)
    except:
        print("Could not load label encoder or feature names for evaluation.")
        le = None
        features = [f"Feature_{i}" for i in range(X_test.shape[2])]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    BATCH_SIZE = 1000
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch = torch.tensor(X_test[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            y_pred.extend(preds)
            y_prob.extend(probs)
        
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    top1 = accuracy_score(y_test, y_pred)
    
    top3_correct = 0
    top5_correct = 0
    
    for i in range(len(y_test)):
        sorted_probs = np.argsort(y_prob[i])
        if y_test[i] in sorted_probs[-3:]:
            top3_correct += 1
        if y_test[i] in sorted_probs[-5:]:
            top5_correct += 1
            
    top3 = top3_correct / len(y_test)
    top5 = top5_correct / len(y_test)
    
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
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    
    mask = np.isin(y_test, top20_classes)
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    
    if len(y_test_sub) > 0:
        cm = confusion_matrix(y_test_sub, y_pred_sub, labels=top20_classes)
        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes, 
                    xticklabels=top20_labels, yticklabels=top20_labels)
        axes.set_title("Confusion Matrix (Top 20 Classes)")
        axes.set_ylabel('True Label')
        axes.set_xlabel('Predicted Label')
        axes.tick_params(axis='x', rotation=90)
        
    plt.tight_layout()
    plt.savefig('evaluation_report.png')
    print(f"[{datetime.now().isoformat()}] Saved evaluation_report.png")
    
    return metrics

if __name__ == "__main__":
    df = pd.read_csv('feature_matrix.csv')
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))
    print(f"Number of classes: {num_classes}")
    
    model = train_lstm(X_train, y_train, input_dim=X_train.shape[2], output_dim=num_classes, epochs=100, batch_size=256)
    metrics = evaluate_model(model, X_test, y_test)
