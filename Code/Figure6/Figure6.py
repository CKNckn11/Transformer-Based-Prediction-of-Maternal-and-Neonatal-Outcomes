import os   
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix  
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, mean_squared_error)
import matplotlib
import shap  

# Configure fonts for Chinese text rendering
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------- 1. Configuration parameters ----------------------
class Config:
    snp_path = r"E:\pregnancy_gwas\test_mid_late\hypothyroidism.csv"        
    mid_blood_path = r"E:\pregnancy_gwas\test_mid_late\mid_pregnancy_blood_profile.csv"  
    late_blood_path = r"E:\pregnancy_gwas\test_mid_late\late_pregnancy_blood_profile.csv" 
    label_path = r"E:\pregnancy_gwas\test_mid_late\hypothyroidism_label.csv"  

    root_output_dir = r"E:\pregnancy_gwas\test_mid_late"  
    output_subdir = "plot"                  

    d_model = 128           
    snp_nhead = 4           
    blood_nhead = 4         
    diff_nhead = 4          
    num_encoder_layers = 2  
    dim_feedforward = 256   
    dropout = 0.3           

    batch_size = 8          
    epochs = 100            
    learning_rate = 1e-4    
    weight_decay = 1e-5     

    n_splits = 5            
    random_state = 47       

    top_k = 10              
    dpi = 300               
    task_type = "classification"  

    shap_sample_size = 50   
    shap_plot_type = "both" 


# ---------------------- 2. Dataset class ----------------------
class QuadModalDataset(Dataset):
    def __init__(self, config, impute_strategy='mean', diff_mode="late-minus-mid"):
        self.snp_path = config.snp_path
        self.mid_blood_path = config.mid_blood_path
        self.late_blood_path = config.late_blood_path
        self.label_path = config.label_path

        try:
            self.snp = pd.read_csv(self.snp_path, index_col=0).T  
            self.mid_blood = pd.read_csv(self.mid_blood_path, index_col=0).T  
            self.late_blood = pd.read_csv(self.late_blood_path, index_col=0).T  
            self.labels = pd.read_csv(self.label_path, index_col=0)  
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

        self.snp.index = self.snp.index.str.strip().str.lower()
        self.mid_blood.index = self.mid_blood.index.str.strip().str.lower()
        self.late_blood.index = self.late_blood.index.str.strip().str.lower()
        self.labels.index = self.labels.index.str.strip().str.lower()

        common_samples = self.snp.index.intersection(
            self.mid_blood.index.intersection(self.late_blood.index.intersection(self.labels.index))
        )
        if len(common_samples) == 0:
            raise ValueError("No common samples were found across all modalities!")

        self.snp = self.snp.loc[common_samples].reset_index(drop=True)
        self.mid_blood = self.mid_blood.loc[common_samples].reset_index(drop=True)
        self.late_blood = self.late_blood.loc[common_samples].reset_index(drop=True)
        self.labels = self.labels.loc[common_samples].values.squeeze()  

        overlap_cols = self.mid_blood.columns.intersection(self.late_blood.columns)
        print(f"Number of automatically matched overlapping mid-to-late blood test indicators: {len(overlap_cols)}")
        if len(overlap_cols) == 0:
            raise ValueError("No overlapping blood test indicators were found between the mid- and late-pregnancy datasets!")

        if diff_mode == "late-minus-mid":
            self.diff_blood = self.late_blood[overlap_cols] - self.mid_blood[overlap_cols]
        else:
            self.diff_blood = self.mid_blood[overlap_cols] - self.late_blood[overlap_cols]

        self.snp, _ = self._process_missing(self.snp, "numerical", impute_strategy)
        self.mid_blood, _ = self._process_missing(self.mid_blood, "numerical", impute_strategy)
        self.late_blood, _ = self._process_missing(self.late_blood, "numerical", impute_strategy)
        self.diff_blood, _ = self._process_missing(self.diff_blood, "numerical", impute_strategy)

        self.scaler_snp = StandardScaler()
        self.scaler_mid = StandardScaler()
        self.scaler_late = StandardScaler()
        self.scaler_diff = StandardScaler()

        self.snp = self.scaler_snp.fit_transform(self.snp)
        self.mid_blood = self.scaler_mid.fit_transform(self.mid_blood)
        self.late_blood = self.scaler_late.fit_transform(self.late_blood)
        self.diff_blood = self.scaler_diff.fit_transform(self.diff_blood)

        self.snp_dim = self.snp.shape[1]
        self.mid_dim = self.mid_blood.shape[1]
        self.late_dim = self.late_blood.shape[1]
        self.diff_dim = self.diff_blood.shape[1]
        print(f"\nFeature dimensions of each modality:")
        print(f"SNP: {self.snp_dim} | Mid-pregnancy blood tests: {self.mid_dim} | Late-pregnancy blood tests: {self.late_dim} | Temporal difference: {self.diff_dim}")
        print(f"Total sample size: {len(self.snp)} | Label distribution: {np.bincount(self.labels.astype(int)) if config.task_type == 'classification' else 'continuous regression values'}")

    def _process_missing(self, data, feat_type, strategy):
        valid_cols = ~data.isna().all(axis=0)
        valid_cols = valid_cols.values
        data = data.iloc[:, valid_cols]
        if strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif strategy == "median":
            imputer = SimpleImputer(strategy="median")
        else:
            raise ValueError(f"Unsupported missing-value imputation strategy: {strategy}")
        data_imputed = imputer.fit_transform(data)
        return data_imputed, valid_cols

    def __len__(self):
        return len(self.snp)

    def __getitem__(self, idx):
        return (
            (
                torch.FloatTensor(self.snp[idx]),          
                torch.FloatTensor(self.mid_blood[idx]),     
                torch.FloatTensor(self.late_blood[idx]),    
                torch.FloatTensor(self.diff_blood[idx])     
            ),
            torch.tensor(self.labels[idx], dtype=torch.float32)  
        )


# ---------------------- 3. Transformer building blocks ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ModalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj(x) * math.sqrt(self.proj.out_features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x.squeeze(1)


# ---------------------- 4. Four-modal fusion model ----------------------
class QuadModalFusionModel(nn.Module):
    def __init__(self, config, snp_dim, mid_dim, late_dim, diff_dim):
        super().__init__()
        self.snp_encoder = ModalTransformerEncoder(
            input_dim=snp_dim,
            d_model=config.d_model,
            nhead=config.snp_nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.mid_encoder = ModalTransformerEncoder(
            input_dim=mid_dim,
            d_model=config.d_model,
            nhead=config.blood_nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.late_encoder = ModalTransformerEncoder(
            input_dim=late_dim,
            d_model=config.d_model,
            nhead=config.blood_nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.diff_encoder = ModalTransformerEncoder(
            input_dim=diff_dim,
            d_model=config.d_model,
            nhead=config.diff_nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model * 4,
            num_heads=config.snp_nhead + config.blood_nhead * 2 + config.diff_nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.BatchNorm1d(config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if not isinstance(x, tuple) or len(x) != 4:
            raise ValueError(f"Model input must be a 4-element tuple; current input: {type(x)} (length: {len(x) if isinstance(x, (tuple, list)) else 'none'})")
        
        snp_feat, mid_feat, late_feat, diff_feat = x
        for name, feat in [("SNP", snp_feat), ("Mid-pregnancy blood tests", mid_feat), ("Late-pregnancy blood tests", late_feat), ("Difference", diff_feat)]:
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f"{name} features must be torch.Tensor; current type: {type(feat)}")
        
        z_snp = self.snp_encoder(snp_feat)
        z_mid = self.mid_encoder(mid_feat)
        z_late = self.late_encoder(late_feat)
        z_diff = self.diff_encoder(diff_feat)
        
        fused_raw = torch.cat([z_snp, z_mid, z_late, z_diff], dim=1)
        fused_raw = fused_raw.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=fused_raw, key=fused_raw, value=fused_raw)
        return self.fusion_head(attn_out.squeeze(1))


# ---------------------- 5. Training and evaluation utilities with PPV and NPV ----------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for (snp, mid, late, diff), labels in loader:
        snp, mid, late, diff = snp.to(device), mid.to(device), late.to(device), diff.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model((snp, mid, late, diff)).squeeze()
        
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)
        
        loss.backward()
        optimizer.step()
        
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    train_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, train_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []
    
    with torch.no_grad():
        for (snp, mid, late, diff), labels in loader:
            snp, mid, late, diff = snp.to(device), mid.to(device), late.to(device), diff.to(device)
            labels = labels.to(device)
            
            outputs = model((snp, mid, late, diff)).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            
            prob = torch.sigmoid(outputs).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            all_prob.extend(prob)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(pred)
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "val_loss": avg_loss,
        "acc": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
        "f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "true_labels": all_true,
        "pred_probs": all_prob
    }
    if len(np.unique(all_true)) == 2:
        metrics["auc"] = roc_auc_score(all_true, all_prob)
    else:
        metrics["auc"] = None
    
    # Calculate the confusion matrix for sensitivity, specificity, PPV, and NPV
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity (recall)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value (PPV)
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value (NPV)
    
    return metrics


# ---------------------- 6. Visualization and result-saving utilities ----------------------
def plot_loss_curves(all_train_losses, all_val_losses, save_path, config):
    plt.figure(figsize=(10, 6), dpi=config.dpi)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    for i, (color, train_loss, val_loss) in enumerate(zip(colors, all_train_losses, all_val_losses)):
        epochs = range(1, len(train_loss) + 1)
        # Plot training loss for each fold with only the first fold labeled to avoid legend clutter
        plt.plot(epochs, train_loss, color=color, linestyle='--', alpha=0.4, linewidth=1.5,
                 label=f'Fold {i+1} Training' if i == 0 else "")
        # Plot validation loss for each fold with only the first fold labeled
        plt.plot(epochs, val_loss, color=color, alpha=0.4, linewidth=1.5,
                 label=f'Fold {i+1} Validation' if i == 0 else "")
    
    # Calculate and plot mean training and validation loss using emphasized lines and markers
    mean_train_loss = np.mean(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    epochs = range(1, len(mean_train_loss) + 1)
    plt.plot(epochs, mean_train_loss, color='navy', linewidth=3, marker='o', markersize=5, markevery=5,
             label='Mean training loss')
    plt.plot(epochs, mean_val_loss, color='darkorange', linewidth=3, marker='s', markersize=5, markevery=5,
             label='Mean validation loss')
    
    # Configure plot formatting
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Four-modal model training-validation loss curve\n(Binary classification task)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xlim(1, len(epochs))  # Match the x-axis range to the training epochs
    # Add a 10% y-axis margin to prevent curves from touching plot boundaries
    plt.ylim(0, max(np.max(all_train_losses), np.max(all_val_losses)) * 1.1)
    
    # Save the figure as a high-resolution PDF and avoid clipped labels
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nLoss curve saved to: {save_path}")


def save_metrics_to_csv(metrics_list, save_path):
    metrics_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(metrics_list))],
        "Validation loss (Val Loss)": [round(m["val_loss"], 4) for m in metrics_list],
        "Accuracy": [round(m["acc"], 4) for m in metrics_list],
        "Precision": [round(m["precision"], 4) for m in metrics_list],
        "Recall": [round(m["recall"], 4) for m in metrics_list],
        "F1 score": [round(m["f1"], 4) for m in metrics_list],
        "Sensitivity（Sensitivity)": [round(m["sensitivity"], 4) for m in metrics_list],
        "Specificity（Specificity)": [round(m["specificity"], 4) for m in metrics_list],
        "Positive predictive value (PPV)": [round(m["ppv"], 4) for m in metrics_list],  # Add PPV
        "Negative predictive value (NPV)": [round(m["npv"], 4) for m in metrics_list],  # Add NPV
        "AUC value": [round(m["auc"], 4) if m["auc"] is not None else "N/A" for m in metrics_list]
    })
    
    # Calculate the mean row
    mean_row = {
        "Fold": "Mean",
        "Validation loss (Val Loss)": round(np.mean([m["val_loss"] for m in metrics_list]), 4),
        "Accuracy": round(np.mean([m["acc"] for m in metrics_list]), 4),
        "Precision": round(np.mean([m["precision"] for m in metrics_list]), 4),
        "Recall": round(np.mean([m["recall"] for m in metrics_list]), 4),
        "F1 score": round(np.mean([m["f1"] for m in metrics_list]), 4),
        "Sensitivity（Sensitivity)": round(np.mean([m["sensitivity"] for m in metrics_list]), 4),
        "Specificity（Specificity)": round(np.mean([m["specificity"] for m in metrics_list]), 4),
        "Positive predictive value (PPV)": round(np.mean([m["ppv"] for m in metrics_list]), 4),  # Add mean PPV
        "Negative predictive value (NPV)": round(np.mean([m["npv"] for m in metrics_list]), 4),  # Add mean NPV
        "AUC value": round(np.mean([m["auc"] for m in metrics_list if m["auc"] is not None]), 4) 
                if any(m["auc"] is not None for m in metrics_list) else "N/A"
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save the CSV file using UTF-8-SIG encoding for Chinese text compatibility
    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Cross-validation metric CSV saved to: {save_path}")


def plot_mean_roc_curve(all_true_labels, all_pred_probs, all_aucs, save_path, config):
    valid_true = []
    valid_prob = []
    valid_aucs = []
    for y_true, y_prob, auc in zip(all_true_labels, all_pred_probs, all_aucs):
        if len(np.unique(y_true)) == 2 and auc is not None:
            valid_true.append(y_true)
            valid_prob.append(y_prob)
            valid_aucs.append(auc)
    
    if not valid_true:
        print("❌ No valid binary classification data are available; ROC curve cannot be plotted")
        return
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for y_true, y_prob in zip(valid_true, valid_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        f = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
        tprs.append(f(mean_fpr))
    
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(valid_aucs)
    std_auc = np.std(valid_aucs)
    
    plt.figure(figsize=(8, 8), dpi=config.dpi)
    plt.plot(mean_fpr, mean_tpr, color='#2E86AB', linewidth=4, alpha=0.9,
             label=f'Mean ROC Curve (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='#A23B72', alpha=0.2, label='±1 Standard Deviation')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=3, color='#F18F01', alpha=0.8,
             label='Random Guess (AUC = 0.5)')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    plt.title(f'5-Fold Cross-Validation Mean ROC Curve\n(Binary classification task)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Mean ROC curve saved to: {save_path}")


def get_feature_importance(model, dataset, config):
    model.eval()
    device = next(model.parameters()).device
    
    feature_importance = {}
    snp_proj_weight = model.snp_encoder.proj.weight.detach().cpu().numpy()
    snp_importance = np.sum(np.abs(snp_proj_weight), axis=0)
    snp_importance = snp_importance / np.max(snp_importance) if np.max(snp_importance) > 0 else np.zeros_like(snp_importance)
    feature_importance["SNP"] = snp_importance
    
    mid_proj_weight = model.mid_encoder.proj.weight.detach().cpu().numpy()
    mid_importance = np.sum(np.abs(mid_proj_weight), axis=0)
    mid_importance = mid_importance / np.max(mid_importance) if np.max(mid_importance) > 0 else np.zeros_like(mid_importance)
    feature_importance["Mid-pregnancy blood tests"] = mid_importance
    
    late_proj_weight = model.late_encoder.proj.weight.detach().cpu().numpy()
    late_importance = np.sum(np.abs(late_proj_weight), axis=0)
    late_importance = late_importance / np.max(late_importance) if np.max(late_importance) > 0 else np.zeros_like(late_importance)
    feature_importance["Late-pregnancy blood tests"] = late_importance
    
    diff_proj_weight = model.diff_encoder.proj.weight.detach().cpu().numpy()
    diff_importance = np.sum(np.abs(diff_proj_weight), axis=0)
    diff_importance = diff_importance / np.max(diff_importance) if np.max(diff_importance) > 0 else np.zeros_like(diff_importance)
    feature_importance["Temporal difference"] = diff_importance
    
    return feature_importance


def plot_accuracy_curves(all_train_accs, all_val_accs, save_path, config):
    plt.figure(figsize=(10, 6), dpi=config.dpi)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    for i, (color, train_acc, val_acc) in enumerate(zip(colors, all_train_accs, all_val_accs)):
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, color=color, linestyle='--', alpha=0.4, linewidth=1.5,
                 label=f'Fold {i+1} Training' if i == 0 else "")
        plt.plot(epochs, val_acc, color=color, alpha=0.4, linewidth=1.5,
                 label=f'Fold {i+1} Validation' if i == 0 else "")
    
    mean_train_acc = np.mean(all_train_accs, axis=0)
    mean_val_acc = np.mean(all_val_accs, axis=0)
    epochs = range(1, len(mean_train_acc) + 1)
    plt.plot(epochs, mean_train_acc, color='navy', linewidth=3, marker='o', markersize=5, markevery=5,
             label='Mean training accuracy')
    plt.plot(epochs, mean_val_acc, color='darkorange', linewidth=3, marker='s', markersize=5, markevery=5,
             label='Mean validation accuracy')
    
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Four-modal model training-validation accuracy curve\n(Binary classification task)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xlim(1, len(epochs))
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Accuracy curve saved to: {save_path}")


def plot_top10_features(importance, modal_name, save_path, config):
    sorted_feat = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features, weights = zip(*sorted_feat)
    
    plt.figure(figsize=(12, 8), dpi=config.dpi)
    colors = ['#2E86AB' if w > 0 else '#E63946' for w in weights]
    bars = plt.barh(features, weights, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', va='center', fontsize=10)
    
    modal_title = {
        "snp": "SNP loci",
        "mid_blood": "Mid-pregnancy blood test indicators",
        "late_blood": "Late-pregnancy blood test indicators",
        "diff_blood": "Temporal difference indicators"
    }
    plt.xlabel('Feature importance (attention weight)', fontsize=14, fontweight='bold')
    plt.title(f'Top 10 influential {modal_title[modal_name]}', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ {modal_title[modal_name]}TOP10 bar plot saved to: {save_path}")


def explain_with_shap(model, dataloader, feature_names, modal_name, save_path, config):
    model.eval()
    device = next(model.parameters()).device
    
    modal_to_idx = {"SNP": 0, "Mid-pregnancy blood tests": 1, "Late-pregnancy blood tests": 2, "Temporal difference": 3}
    if modal_name not in modal_to_idx:
        raise ValueError(f"Unsupported modality name: {modal_name}")
    modal_idx = modal_to_idx[modal_name]
    
    all_samples = []
    total_samples = 0
    for batch in dataloader:
        (snp, mid, late, diff), _ = batch
        batch_size = snp.shape[0]
        if total_samples + batch_size > config.shap_sample_size:
            need = config.shap_sample_size - total_samples
            snp = snp[:need]
            mid = mid[:need]
            late = late[:need]
            diff = diff[:need]
            all_samples.append((snp, mid, late, diff))
            total_samples = config.shap_sample_size
            break
        all_samples.append((snp, mid, late, diff))
        total_samples += batch_size
        if total_samples >= config.shap_sample_size:
            break
    
    if total_samples == 0:
        raise RuntimeError(f"No available samples for SHAP analysis")
    print(f"⚠️ Sample size used for SHAP analysis: {total_samples}")
    
    snp_samples = torch.cat([x[0] for x in all_samples]).to(device)
    mid_samples = torch.cat([x[1] for x in all_samples]).to(device)
    late_samples = torch.cat([x[2] for x in all_samples]).to(device)
    diff_samples = torch.cat([x[3] for x in all_samples]).to(device)
    
    target_sample_size = min(total_samples, config.shap_sample_size)
    snp_samples = snp_samples[:target_sample_size]
    mid_samples = mid_samples[:target_sample_size]
    late_samples = late_samples[:target_sample_size]
    diff_samples = diff_samples[:target_sample_size]
    
    fixed_data = (snp_samples, mid_samples, late_samples, diff_samples)
    
    class SingleModalWrapper(nn.Module):
        def __init__(self, base_model, fixed_data, target_modal_idx):
            super().__init__()
            self.base_model = base_model
            self.fixed_data = fixed_data
            self.target_modal_idx = target_modal_idx
        
        def forward(self, x):
            x = x.requires_grad_(True)
            fixed_sample_size = self.fixed_data[0].shape[0]
            if x.shape[0] != fixed_sample_size:
                x = x[:fixed_sample_size]
            full_input_list = list(self.fixed_data)
            full_input_list[self.target_modal_idx] = x.to(device)
            full_input = tuple(full_input_list)
            logits = self.base_model(full_input)
            probs = torch.sigmoid(logits)
            return probs
    
    wrapper_model = SingleModalWrapper(
        base_model=model,
        fixed_data=fixed_data,
        target_modal_idx=modal_idx
    ).to(device)
    wrapper_model.eval()
    
    current_modal_data = fixed_data[modal_idx].cpu().numpy()
    current_modal_data = current_modal_data[:target_sample_size]
    
    x_data = torch.FloatTensor(current_modal_data).to(device)
    x_data = x_data.requires_grad_(True)
    explainer = shap.GradientExplainer(
        model=wrapper_model,
        data=x_data,
        local_smoothing=0.1
    )
    
    x_input = torch.FloatTensor(current_modal_data).to(device)[:target_sample_size]
    x_input = x_input.requires_grad_(True)
    shap_values = explainer.shap_values(x_input)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    shap_values = np.squeeze(shap_values)[:target_sample_size]
    
    valid_feat_num = min(len(feature_names), shap_values.shape[1]) if shap_values.ndim > 1 else len(feature_names)
    plot_feature_names = feature_names[:valid_feat_num]
    shap_vals_for_plot = shap_values[:, :valid_feat_num]
    
    if config.shap_plot_type in ["both", "bar"]:
        plt.figure(figsize=(12, 8), dpi=config.dpi)
        feat_shap_sum = np.sum(shap_vals_for_plot, axis=0)
        top_k_idx = np.argsort(np.abs(feat_shap_sum))[-config.top_k:][::-1]
        top_k_names = [plot_feature_names[i] for i in top_k_idx]
        top_k_sums = feat_shap_sum[top_k_idx]
        
        colors = ['#2E86AB' if s > 0 else '#E63946' for s in top_k_sums]
        bars = plt.barh(top_k_names, top_k_sums, color=colors)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + (0.001 if width > 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f'{width:.6f}',
                va='center', 
                fontsize=10
            )
        
        plt.xlabel('Summed SHAP value (positive values promote the positive class; negative values suppress it)', fontsize=14, fontweight='bold')
        plt.title(f'{modal_name} feature summed SHAP impact top {config.top_k}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, f'{modal_name}_shap_sum_bar.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✅ {modal_name}  summed SHAP bar plot saved")
    
    if config.shap_plot_type in ["both", "beeswarm"]:
        plt.figure(figsize=(12, 8), dpi=config.dpi)
        shap.summary_plot(
            shap_vals_for_plot,
            current_modal_data[:, :valid_feat_num],
            feature_names=plot_feature_names,
            show=False,
            max_display=config.top_k
        )
        plt.title(f'{modal_name} feature single-sample SHAP distribution (positive and negative effects)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('SHAP value (positive values promote the positive class; negative values suppress it)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{modal_name}_shap_beeswarm.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✅ {modal_name}  SHAP beeswarm plot saved")
    
    shap_sum_df = pd.DataFrame(
        data=[np.sum(shap_vals_for_plot, axis=0)],
        columns=[f'{modal_name}_{feat}' for feat in plot_feature_names]
    )
    shap_sum_df.to_csv(os.path.join(save_path, f'{modal_name}_shap_sum_values.csv'), index=False, encoding='utf-8-sig')
    
    shap_raw_df = pd.DataFrame(
        shap_vals_for_plot,
        columns=[f'{modal_name}_{feat}' for feat in plot_feature_names]
    )
    shap_raw_df.to_csv(os.path.join(save_path, f'{modal_name}_shap_raw_values.csv'), index=False, encoding='utf-8-sig')
    
    print(f"✅ {modal_name}  summed SHAP analysis completed")
    return shap_values


def plot_mean_shap_bar(mean_shap_df, modal_name, save_path, top_k=10):
    feature_cols = mean_shap_df.columns
    feature_names = [col.split(f"{modal_name}_")[-1] for col in feature_cols]
    shap_means = mean_shap_df.values[0]
    
    sorted_indices = np.argsort(np.abs(shap_means))[-top_k:][::-1]
    top_names = [feature_names[i] for i in sorted_indices]
    top_means = shap_means[sorted_indices]
    
    plt.figure(figsize=(12, 8), dpi=150)
    colors = ['#E63946' if val > 0 else '#2E86AB' for val in top_means]
    bars = plt.barh(top_names, top_means, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        text_x = width + 0.0005 if width > 0 else width - 0.0005
        plt.text(
            text_x, 
            bar.get_y() + bar.get_height()/2,
            f"{width:.6f}",
            va='center', 
            fontsize=10
        )
    
    plt.xlabel(f'Mean summed SHAP value across 5 folds (positive values promote the positive class; negative values suppress it)', fontsize=14, fontweight='bold')
    plt.title(f'{modal_name} features - top mean summed SHAP values across 5-fold cross-validation {top_k}', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {modal_name} SHAP mean plot: {save_path}")


# ---------------------- Performance metric scatter plot function with PPV and NPV ----------------------
def plot_performance_metrics_scatter(disease_name, disease_output_dir, results, config):
    metrics_scatter_filename = f"{disease_name}_performance_metrics_scatter.pdf"
    full_save_path = os.path.join(disease_output_dir, metrics_scatter_filename)

    labels = [f'Fold {i+1}' for i in range(len(results))]
    
    # Add PPV and NPV to the scatter plot data
    metrics_data = {}
    for i, label in enumerate(labels):
        metrics_data[label] = [
            results[i]['precision'],
            results[i]['recall'],
            results[i]['f1'],
            results[i]['ppv'],  # Add PPV
            results[i]['npv'],  # Add NPV
            results[i]['auc'] if results[i]['auc'] is not None else 0
        ]
    
    # Update metric names accordingly
    metrics = ['Precision', 'Recall', 'F1 Score', 
               'PPV', 'NPV', 'AUC Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(14, 8), dpi=config.dpi)  # Enlarge the figure to accommodate the added metrics
    x_pos = np.arange(len(metrics))
    width = 0.12  # Adjust point spacing to avoid overlap

    # Plot metric scatter points for each fold
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = metrics_data[label]
        plt.scatter(
            x_pos + i * width,
            values,
            color=color,
            s=120,
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            label=label
        )
    
    # Calculate and plot mean lines
    all_values = [
        [res['precision'], res['recall'], res['f1'], res['ppv'], res['npv'], 
         res['auc'] if res['auc'] is not None else 0]
        for res in results
    ]
    mean_values = np.mean(all_values, axis=0)
    plt.plot(
        x_pos + width * (len(labels) - 1) / 2,
        mean_values,
        color='red',
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=10,
        label='Mean'
    )

    # Configure plot formatting
    plt.xticks(x_pos + width * (len(labels) - 1) / 2, metrics, fontsize=11, rotation=15)  # Rotate x-axis labels to avoid overlap
    plt.ylim(-0.05, 1.05)  # Classification metrics range from 0 to 1
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    # Place the legend at the top with horizontal layout
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
              ncol=config.n_splits + 1, fontsize=10, frameon=True)

    plt.title(f'Performance Metrics Across Cross-Validation Folds\nfor {disease_name} with PPV/NPV', 
             fontsize=16, fontweight='bold', pad=30)
    plt.ylabel('Score value', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(full_save_path, format='pdf', dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"Performance metric scatter plot with PPV/NPV saved to: {full_save_path}")

    return {
        'metrics': metrics,
        'mean': mean_values,
        'std': np.std(all_values, axis=0)
    }


# ---------------------- Sensitivity and specificity helper functions ----------------------
def calculate_sensitivity_specificity(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def plot_sensitivity_curves(all_sensitivities, save_path, config):
    plt.figure(figsize=(10, 6), dpi=150)
    epochs = range(1, len(all_sensitivities[0]) + 1)
    for i, sens in enumerate(all_sensitivities):
        plt.plot(epochs, sens, linestyle='--', alpha=0.6, label=f'fold {i+1}')
    mean_sens = np.mean(all_sensitivities, axis=0)
    plt.plot(epochs, mean_sens, color='red', linewidth=2, label='Mean')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Sensitivity（Sensitivity)', fontsize=12)
    plt.title(f'{config.n_splits}-fold cross-validation - validation sensitivity curve', fontsize=14, pad=15)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Sensitivity curve saved: {save_path}")


def plot_specificity_curves(all_specificities, save_path, config):
    plt.figure(figsize=(10, 6), dpi=150)
    epochs = range(1, len(all_specificities[0]) + 1)
    for i, spec in enumerate(all_specificities):
        plt.plot(epochs, spec, linestyle='--', alpha=0.6, label=f'fold {i+1}')
    mean_spec = np.mean(all_specificities, axis=0)
    plt.plot(epochs, mean_spec, color='blue', linewidth=2, label='Mean')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Specificity（Specificity)', fontsize=12)
    plt.title(f'{config.n_splits}-fold cross-validation - validation specificity curve', fontsize=14, pad=15)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Specificity curve saved: {save_path}")


def plot_sens_spec_comparison(all_sensitivities, all_specificities, save_path, config):
    final_sens = [sens[-1] for sens in all_sensitivities]
    final_spec = [spec[-1] for spec in all_specificities]
    mean_sens = np.mean(final_sens)
    mean_spec = np.mean(final_spec)
    std_sens = np.std(final_sens)
    std_spec = np.std(final_spec)
    
    plt.figure(figsize=(8, 6), dpi=150)
    x = np.arange(2)
    bars = plt.bar(x, [mean_sens, mean_spec], yerr=[std_sens, std_spec], 
                   capsize=10, color=['#FF6B6B', '#4ECDC4'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f} ± {std_sens:.4f}' if bar == bars[0] else f'{height:.4f} ± {std_spec:.4f}',
                 ha='center', fontsize=10)
    plt.xticks(x, ['Sensitivity（Sensitivity)', 'Specificity（Specificity)'])
    plt.ylabel('Metric value (mean +/- SD at the last epoch)', fontsize=12)
    plt.title(f'{config.n_splits}-fold cross-validation - sensitivity vs specificity', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Sensitivity vs specificity comparison plot saved: {save_path}")


# ---------------------- Main training workflow ----------------------
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"Current device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Task type: {config.task_type.capitalize()}(binary classification)")
    print(f"{'='*60}")
    
    required_files = [config.snp_path, config.mid_blood_path, config.late_blood_path, config.label_path]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file does not exist: {file}")
    print("All input files passed validation ✅")
    
    try:
        dataset = QuadModalDataset(
            config=config,
            impute_strategy="mean",
            diff_mode="late-minus-mid"
        )
    except Exception as e:
        raise RuntimeError(f"Dataset initialization failed: {str(e)}")
    print("Four-modal dataset loaded successfully ✅")
    

    min_train_samples = len(dataset) * (config.n_splits - 1) // config.n_splits
    if config.batch_size >= min_train_samples:
        raise ValueError(f"batch_size={config.batch_size}  is too large; the smallest training fold contains only {min_train_samples} samples")
    

    output_dir = os.path.join(config.root_output_dir, config.output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    all_metrics = []
    all_true_labels = []
    all_pred_probs = []
    all_aucs = []
    all_sensitivities = []
    all_specificities = []
    
    all_feature_importance = {
        "SNP": [],
        "Mid-pregnancy blood tests": [],
        "Late-pregnancy blood tests": [],
        "Temporal difference": []
    }
    
    shap_sum_all_folds = {
        "SNP": [],
        "Mid-pregnancy blood tests": [],
        "Late-pregnancy blood tests": [],
        "Temporal difference": []
    }
    
    snp_df = pd.read_csv(config.snp_path, index_col=0).T
    snp_df.index = snp_df.index.str.strip().str.lower()
    common_samples = snp_df.index.intersection(
        pd.read_csv(config.mid_blood_path, index_col=0).T.index.str.strip().str.lower().intersection(
            pd.read_csv(config.late_blood_path, index_col=0).T.index.str.strip().str.lower().intersection(
                pd.read_csv(config.label_path, index_col=0).index.str.strip().str.lower()
            )
        )
    )
    snp_df = snp_df.loc[common_samples]
    _, snp_valid_cols = dataset._process_missing(snp_df, "numerical", "mean")
    snp_names = snp_df.columns[snp_valid_cols].tolist()
    
    mid_df = pd.read_csv(config.mid_blood_path, index_col=0).T
    mid_df.index = mid_df.index.str.strip().str.lower()
    mid_df = mid_df.loc[common_samples]
    _, mid_valid_cols = dataset._process_missing(mid_df, "numerical", "mean")
    mid_names = mid_df.columns[mid_valid_cols].tolist()
    
    late_df = pd.read_csv(config.late_blood_path, index_col=0).T
    late_df.index = late_df.index.str.strip().str.lower()
    late_df = late_df.loc[common_samples]
    _, late_valid_cols = dataset._process_missing(late_df, "numerical", "mean")
    late_names = late_df.columns[late_valid_cols].tolist()
    
    overlap_cols = mid_df.columns.intersection(late_df.columns)
    diff_names = overlap_cols.tolist()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset.snp, dataset.labels), 1):
        print(f"\n{'='*60}")
        print(f"Starting fold {fold}/{config.n_splits} of cross-validation")
        print(f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        current_model = QuadModalFusionModel(
            config=config,
            snp_dim=dataset.snp_dim,
            mid_dim=dataset.mid_dim,
            late_dim=dataset.late_dim,
            diff_dim=dataset.diff_dim
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            current_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        fold_val_accs = []
        fold_sensitivities = []
        fold_specificities = []
        best_val_auc = -1
        best_model_weights = None
        
        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = train_epoch(current_model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(current_model, val_loader, criterion, device)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_metrics["val_loss"])
            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_metrics["acc"])
            fold_sensitivities.append(val_metrics["sensitivity"])
            fold_specificities.append(val_metrics["specificity"])
            
            if val_metrics["auc"] is not None and val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_model_weights = current_model.state_dict()
                print(f"Epoch {epoch} updated the best model (Val AUC: {best_val_auc:.4f})")
            
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.4f} | "
                  f"Val Sensitivity: {val_metrics['sensitivity']:.4f} | "
                  f"Val Specificity: {val_metrics['specificity']:.4f} | "
                  f"Val PPV: {val_metrics['ppv']:.4f} | "  # Print PPV
                  f"Val NPV: {val_metrics['npv']:.4f} | "  # Print NPV
                  f"Val AUC: {val_metrics['auc']:.4f}" if val_metrics["auc"] is not None else "")
        
        best_model = QuadModalFusionModel(
            config=config,
            snp_dim=dataset.snp_dim,
            mid_dim=dataset.mid_dim,
            late_dim=dataset.late_dim,
            diff_dim=dataset.diff_dim
        ).to(device)
        best_model.load_state_dict(best_model_weights)
        
        final_val_metrics = evaluate(best_model, val_loader, criterion, device)
        all_metrics.append(final_val_metrics)
        all_true_labels.append(final_val_metrics["true_labels"])
        all_pred_probs.append(final_val_metrics["pred_probs"])
        all_aucs.append(final_val_metrics["auc"])
        all_sensitivities.append(fold_sensitivities)
        all_specificities.append(fold_specificities)
        
        feat_importance = get_feature_importance(best_model, dataset, config)
        for key in all_feature_importance:
            all_feature_importance[key].append(feat_importance[key])
        
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        modal_feature_names = {
            "SNP": snp_names,
            "Mid-pregnancy blood tests": mid_names,
            "Late-pregnancy blood tests": late_names,
            "Temporal difference": diff_names
        }
        
        shap_loader = DataLoader(
            val_subset, 
            batch_size=min(config.batch_size, len(val_subset)),
            shuffle=False,
            num_workers=0
        )
        
        for modal, names in modal_feature_names.items():
            if len(names) == 0:
                print(f"⚠️ {modal} has no valid features; skipping SHAP analysis")
                continue
            try:
                explain_with_shap(
                    model=best_model,
                    dataloader=shap_loader,
                    feature_names=names,
                    modal_name=modal,
                    save_path=fold_output_dir,
                    config=config
                )
                shap_sum_path = os.path.join(fold_output_dir, f"{modal}_shap_sum_values.csv")
                if os.path.exists(shap_sum_path):
                    df = pd.read_csv(shap_sum_path)
                    shap_sum_all_folds[modal].append(df)
                    print(f"✅ Collected {modal} summed SHAP results (fold {fold})")
                else:
                    print(f"⚠️ Could not find {modal} summed SHAP file (fold {fold})")
            except Exception as e:
                print(f"❌ {modal}  SHAP analysis failed (fold {fold})：{str(e)}")
        
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_train_accs.append(fold_train_accs)
        all_val_accs.append(fold_val_accs)
        
        print(f"Fold {fold} of cross-validationcompleted ✅")
    
    print("\nGenerating mean summed SHAP plots across 5 folds...")
    for modal in shap_sum_all_folds:
        valid_folds = len(shap_sum_all_folds[modal])
        if valid_folds < 3:
            print(f"⚠️ {modal} has insufficient valid folds; skipping mean plot")
            continue
        
        merged_df = pd.concat(shap_sum_all_folds[modal], ignore_index=True)
        mean_shap_df = merged_df.mean(axis=0).to_frame().T
        mean_csv_path = os.path.join(output_dir, f"{modal}_shap_sum_mean.csv")
        mean_shap_df.to_csv(mean_csv_path, index=False, encoding='utf-8-sig')
        
        plot_mean_shap_bar(
            mean_shap_df=mean_shap_df,
            modal_name=modal,
            save_path=os.path.join(output_dir, f"{modal}_shap_sum_mean_bar.pdf"),
            top_k=config.top_k
        )
    
    # Save the metric CSV file with PPV and NPV
    metrics_save_path = os.path.join(output_dir, "cross_validation_metrics_summary.csv")
    save_metrics_to_csv(all_metrics, metrics_save_path)
    
    # Plot all result curves
    loss_curve_path = os.path.join(output_dir, "training_validation_loss_curve.pdf")
    plot_loss_curves(all_train_losses, all_val_losses, loss_curve_path, config)
    
    acc_curve_path = os.path.join(output_dir, "training_validation_accuracy_curve.pdf")
    plot_accuracy_curves(all_train_accs, all_val_accs, acc_curve_path, config)
    
    sensitivity_curve_path = os.path.join(output_dir, "validation_sensitivity_curve.pdf")
    plot_sensitivity_curves(all_sensitivities, sensitivity_curve_path, config)
    
    specificity_curve_path = os.path.join(output_dir, "validation_specificity_curve.pdf")
    plot_specificity_curves(all_specificities, specificity_curve_path, config)
    
    sens_spec_compare_path = os.path.join(output_dir, "sensitivity_vs_specificity_mean_comparison.pdf")
    plot_sens_spec_comparison(all_sensitivities, all_specificities, sens_spec_compare_path, config)
    
    roc_curve_path = os.path.join(output_dir, "five_fold_cross_validation_mean_ROC_curve.pdf")
    plot_mean_roc_curve(all_true_labels, all_pred_probs, all_aucs, roc_curve_path, config)

    # Plot performance metric scatter points with PPV and NPV
    disease_name = "disease"
    plot_performance_metrics_scatter(
        disease_name=disease_name,
        disease_output_dir=output_dir,
        results=all_metrics,
        config=config
    )

    print(f"\n{'='*60}")
    print(f"All cross-validation completed! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    config = Config()
    main(config)