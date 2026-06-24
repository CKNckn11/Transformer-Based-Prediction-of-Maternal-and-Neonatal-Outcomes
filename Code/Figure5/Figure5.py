import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from scipy.interpolate import interp1d
from scipy import stats
import shap  # SHAPtext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# text（text，text）
def set_chinese_font():
    try:
        # Windowstext
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi']
        # Mactext（textMac，text）
        # plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'Arial Unicode MS']
        # Linuxtext（textLinux，text）
        # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # text
    except Exception as e:
        print(f"text，text: {e}")
        plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

# text（text）
set_chinese_font()

# ---------------------- Path and Parameter Configuration ----------------------
class Config:
    # text（text）
    root_data_dir = r"E:\textgwas\text"
    blood_path = r"E:\textgwas\text\text.csv"  # text

    # text
    output_subdir = "plot"  # text

    # text
    d_model = 128
    nhead = 8
    dim_feedforward = 256
    num_encoder_layers = 3
    batch_size = 16
    epochs = 10 
    learning_rate = 5e-5
    weight_decay = 1e-5

    # text
    n_splits = 5
    random_state = 42

    # text  
    top_k = 10  
    dpi = 300   
    shap_nsamples = 100  # SHAPtext
    shap_sample_size = 200  # textSHAPtext
    shap_plot_type = "both"  # text: "both", "bar", "beeswarm"

    # text
    risk_quantiles = [0.45, 0.55, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0]  # text
    reference_interval = "45-55% (Reference)"  # text
    or_values = [  # textORtext（text，text）
        "Reference", 
        "1.19 [1.13-1.26]", "1.34 [1.26-1.43]", "1.56 [1.45-1.67]", "1.95 [1.81-2.10]",
        "2.38 [2.20-2.57]", "2.38 [2.10-2.69]", "2.38 [2.00-2.83]", "2.38 [1.61-3.50]"
    ]


# text
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- Dataset Class ----------------------
class MultiModalDataset(Dataset):
    """text（textSNPtext）"""
    def __init__(self, config, disease_name, impute_strategy='mean', save_scalers=True):
        # 1. text
        self.snp_path = os.path.join(config.root_data_dir, disease_name, f"{disease_name}.csv")
        self.label_path = os.path.join(config.root_data_dir, disease_name, f"{disease_name}_label.csv")
        self.blood_path = config.blood_path

        # 2. text
        self._check_file_exists()

        # 3. text（text，text）
        self.snp_raw = pd.read_csv(self.snp_path, index_col=0).T  # SNPtext (text×text)
        self.blood_raw = pd.read_csv(self.blood_path, index_col=0).T  # text (text×text)
        self.labels_raw = pd.read_csv(self.label_path, index_col=0)  # text（0/1）

        # 4. text（text、text，text）
        self._standardize_indices()

        # 5. text（SNP、text、text）
        self.common_samples = self._get_common_samples()
        if len(self.common_samples) == 0:
            raise ValueError(f"text {disease_name} text，text")

        # 6. text
        self.snp = self.snp_raw.loc[self.common_samples].values  # (n_samples, textSNPtext)
        self.blood = self.blood_raw.loc[self.common_samples].values  # (n_samples, text)
        self.y = self.labels_raw.loc[self.common_samples, self.labels_raw.columns[0]].values.astype(int)

        # 7. text
        self.snp_names_original = self.snp_raw.columns.tolist()
        self.blood_names_original = self.blood_raw.columns.tolist()

        # 8. text（text+text）
        self.snp_imputed, self.snp_valid_mask = self._process_missing_values(
            self.snp, feature_type='numerical', strategy=impute_strategy
        )
        self.blood_imputed, self.blood_valid_mask = self._process_missing_values(
            self.blood, feature_type='numerical', strategy=impute_strategy
        )

        # 9. text（text）
        self.valid_snp_names = [
            self.snp_names_original[i] for i, is_valid in enumerate(self.snp_valid_mask) if is_valid
        ]
        self.valid_blood_names = [
            self.blood_names_original[i] for i, is_valid in enumerate(self.blood_valid_mask) if is_valid
        ]

        # 10. text（text：text）
        self.snp_dim = self.snp_imputed.shape[1]  # textSNPtext = text
        self.blood_dim = self.blood_imputed.shape[1]  # text = text

        # 11. text（text）
        assert self.snp_imputed.shape[1] == len(self.valid_snp_names), \
            f"SNPtext：text{self.snp_imputed.shape[1]} vs text{len(self.valid_snp_names)}"
        assert self.blood_imputed.shape[1] == len(self.valid_blood_names), \
            f"text：text{self.blood_imputed.shape[1]} vs text{len(self.valid_blood_names)}"
        
        # text
        print(f"textSNPtext: {self.snp_imputed.shape} (text×text)")
        print(f"text: {self.blood_imputed.shape} (text×text)")
        print(f"textSNPtext: {self.snp_dim}, text: {self.blood_dim}")

        # 12. text
        self.scaler_snp = StandardScaler()
        self.scaler_blood = StandardScaler()
        self.X_snp = self.scaler_snp.fit_transform(self.snp_imputed)  # (n_samples, snp_dim)
        self.X_blood = self.scaler_blood.fit_transform(self.blood_imputed)  # (n_samples, blood_dim)

        # 13. text
        if save_scalers:
            self._save_scalers(config)

        # 14. text
        self.features = (self.X_snp, self.X_blood)
        self.X = self.features

        # 15. text
        self._print_dataset_info(disease_name)

    def _check_file_exists(self):
        missing_files = []
        for path in [self.snp_path, self.label_path, self.blood_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        if missing_files:
            raise FileNotFoundError(f"text: {', '.join(missing_files)}")

    def _standardize_indices(self):
        self.snp_raw.index = self.snp_raw.index.str.strip().str.lower()
        self.blood_raw.index = self.blood_raw.index.str.strip().str.lower()
        self.labels_raw.index = self.labels_raw.index.str.strip().str.lower()

    def _get_common_samples(self):
        snp_samples = set(self.snp_raw.index)
        blood_samples = set(self.blood_raw.index)
        label_samples = set(self.labels_raw.index)
        return list(snp_samples & blood_samples & label_samples)

    def _process_missing_values(self, data, feature_type='numerical', strategy='mean'):
        valid_mask = ~np.isnan(data).all(axis=0)  # text
        valid_data = data[:, valid_mask]
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif feature_type == 'categorical':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            raise ValueError(f"text: {strategy}")
        imputed_data = imputer.fit_transform(valid_data)
        return imputed_data, valid_mask

    def _save_scalers(self, config):
        scaler_dir = os.path.join(config.root_data_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        import pickle
        with open(os.path.join(scaler_dir, "snp_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler_snp, f)
        with open(os.path.join(scaler_dir, "blood_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler_blood, f)
        print(f"text: {scaler_dir}")

    def _print_dataset_info(self, disease_name):
        n_samples = len(self)
        class_dist = np.bincount(self.y)
        print(f"text {disease_name} text:")
        print(f"  text: {n_samples}")
        print(f"  text: {class_dist} (text {np.unique(self.y)})")
        print(f"  textSNPtext: {self.snp_dim}")
        print(f"  text: {self.blood_dim}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """text（(snp_dim,) text (blood_dim,)）"""
        snp_feat = torch.FloatTensor(self.features[0][idx])  # text: (snp_dim,)
        blood_feat = torch.FloatTensor(self.features[1][idx])# text: (blood_dim,)
        
        # text：text
        assert snp_feat.shape[0] == self.snp_dim, f"SNPtext：text{len(snp_feat)}text，text{self.snp_dim}text"
        assert blood_feat.shape[0] == self.blood_dim, f"text：text{len(blood_feat)}text，text{self.blood_dim}text"
        
        return (
            snp_feat,
            blood_feat,
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# text
class DatasetFromIndices(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# ---------------------- Transformer Components ----------------------
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


class ModalitiesTransformerEncoder(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        x = self.projection(x) * math.sqrt(self.projection.out_features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        x = torch.mean(x, dim=1)
        return x


class CrossModalitiesAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # text：text
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # text：textd_modeltextnheadtext（text）
        assert d_model % nhead == 0, f"d_model({d_model})textnhead({nhead})text"

    def forward(self, query, key_value):
        # text [batch_size, 1, d_model] text（text1）
        query_seq = query.unsqueeze(1)  # text: [batch_size, 1, d_model]
        key_value_seq = key_value.unsqueeze(1)  # text: [batch_size, 1, d_model]
        
        # text
        assert query_seq.shape[2] == self.multihead_attn.embed_dim, "text"
        assert key_value_seq.shape[2] == self.multihead_attn.embed_dim, "text"
        
        attn_output, _ = self.multihead_attn(query_seq, key_value_seq, key_value_seq)
        out = query + self.dropout(attn_output.squeeze(1))  # text
        out = self.norm(out)
        return out


# ---------------------- Multi-modal Fusion Model ----------------------
class MultiModalTransformerModel(nn.Module):
    def __init__(self, snp_dim, blood_dim, d_model=128, nhead=8, 
                 dim_feedforward=256, num_encoder_layers=3, num_classes=2):
        super().__init__()
        self.snp_transformer = ModalitiesTransformerEncoder(
            feature_dim=snp_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers
        )
        self.blood_transformer = ModalitiesTransformerEncoder(
            feature_dim=blood_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers
        )

        self.snp_to_blood_attn = CrossModalitiesAttention(d_model)
        self.blood_to_snp_attn = CrossModalitiesAttention(d_model)

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, snp_features, blood_features):
        # text [batch_size, feature_dim] text
        # text [batch_size, 1, feature_dim] textTransformertext
        snp_reshaped = snp_features.unsqueeze(1)  # text（text=1）
        blood_reshaped = blood_features.unsqueeze(1)
        
        # text
        assert snp_reshaped.shape[2] == self.snp_transformer.projection.in_features, \
            f"SNPtext：text{snp_reshaped.shape[2]}, text{self.snp_transformer.projection.in_features}"
        assert blood_reshaped.shape[2] == self.blood_transformer.projection.in_features, \
            f"text：text{blood_reshaped.shape[2]}, text{self.blood_transformer.projection.in_features}"
        
        z_snp = self.snp_transformer(snp_reshaped)
        z_blood = self.blood_transformer(blood_reshaped)
        
        z_snp_enhanced = self.blood_to_snp_attn(z_snp, z_blood)
        z_blood_enhanced = self.snp_to_blood_attn(z_blood, z_snp)
        
        fused_features = torch.cat([z_snp_enhanced, z_blood_enhanced], dim=1)
        output = self.fusion(fused_features)
        
        return output


# ---------------------- Training and Evaluation Functions ----------------------
def train_epoch(model, loader, optimizer, criterion, device):
    """textepoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # text：SNPtext、text、text
        snp_features, blood_features, y = batch
        snp_features = snp_features.to(device)
        blood_features = blood_features.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(snp_features, blood_features)  # text
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += len(y)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    """textepoch"""
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []
    val_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            # text
            snp_features, blood_features, y = batch
            snp_features = snp_features.to(device)
            blood_features = blood_features.to(device)
            y = y.to(device)

            outputs = model(snp_features, blood_features)
            loss = criterion(outputs, y)
            val_loss += loss.item() * len(y)

            probs = torch.softmax(outputs, dim=1)
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            pred_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    auc_score = roc_auc_score(true_labels, pred_probs) if len(np.unique(true_labels)) == 2 else None
    avg_val_loss = val_loss / len(loader.dataset)

    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc_score,
        'true_labels': true_labels, 'pred_probs': pred_probs, 'val_loss': avg_val_loss
    }


# ---------------------- text：text ----------------------
def predict_sample_risk(model, dataset, device, config):
    """text（text）text"""
    model.eval()
    all_risk_probs = []
    all_sample_ids = dataset.common_samples  # textID
    all_true_labels = dataset.y  # text
    
    # text
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            snp_feat, blood_feat, _ = batch
            snp_feat = snp_feat.to(device)
            blood_feat = blood_feat.to(device)
            
            outputs = model(snp_feat, blood_feat)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # text（text）
            all_risk_probs.extend(probs)
    
    # textDataFrametext（Z-score）
    risk_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'true_label': all_true_labels,
        'risk_prob': all_risk_probs,
        'risk_zscore': stats.zscore(all_risk_probs)  # Z-scoretext
    })
    
    return risk_df


def plot_risk_distribution(risk_df, disease_name, save_dir, config):
    """text（text）- text"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{disease_name}_risk_distribution.pdf")
    
    # text（text）
    quantiles = config.risk_quantiles
    # text（textz-score）
    interval_edges = np.quantile(risk_df['risk_zscore'], quantiles)
    # text：text，text
    interval_edges = np.unique(interval_edges)
    # text（-text），text = text + 1
    full_edges = np.concatenate([[-np.inf], interval_edges])
    
    # text（text = text - 1）
    interval_labels = [
        "45-55% (Reference)", 
        "55-80%", "80-90%", "90-95%", "95-97.5%", 
        "97.5-99%", "99-99.5%", "99.5-99.9%", "99.9-100%"
    ]
    # text，text = text - 1
    num_valid_intervals = len(full_edges) - 1
    interval_labels = interval_labels[:num_valid_intervals]  # text
    # text，text
    while len(interval_labels) < num_valid_intervals:
        interval_labels.append(f"Interval {len(interval_labels)+1}")
    
    # text（text）
    risk_df['risk_interval'] = pd.cut(
        risk_df['risk_zscore'], 
        bins=full_edges, 
        labels=interval_labels, 
        include_lowest=True
    )
    
    # textORtext（textORtext）
    interval_or = dict(zip(interval_labels, config.or_values[:len(interval_labels)]))
    # textORtext，text
    while len(interval_or) < len(interval_labels):
        interval_or[interval_labels[len(interval_or)]] = "N/A"
    
    # text
    plt.figure(figsize=(14, 8), dpi=config.dpi)
    sns.set_style("whitegrid")
    
    # text（text，text）
    colors = sns.color_palette("Greens", n_colors=len(interval_labels))
    if len(interval_labels) >= 1:
        colors[0] = (0.7, 0.7, 0.7, 0.6)  # text（text）
    
    # text
    for i, (interval, color) in enumerate(zip(interval_labels, colors)):
        subset = risk_df[risk_df['risk_interval'] == interval]
        if len(subset) == 0:
            continue  # text
        
        sns.kdeplot(
            subset['risk_zscore'],
            fill=True,
            color=color,
            alpha=0.8,
            linewidth=1.5,
            label=f"{interval}\nOR = {interval_or[interval]}"
        )
    
    # text
    plt.xlabel("Predicted Disease Risk (Z-score Standardized)", fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")
    plt.title(f"Distribution of Predicted Disease Risk for {disease_name}", 
              fontsize=16, fontweight="bold", pad=20)
    
    # text（text）
    plt.legend(
        title="Risk Percentile Interval", 
        loc="upper right", 
        fontsize=10,
        bbox_to_anchor=(1.3, 1),
        frameon=True,
        shadow=True
    )
    
    # text
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ text：{save_path}")
    
    # textCSV
    risk_csv_path = os.path.join(save_dir, f"{disease_name}_sample_risk_scores.csv")
    risk_df.to_csv(risk_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ text：{risk_csv_path}")
    
    return risk_df


# ---------------------- TOP10 SNP vs Others Visualization ----------------------
def calculate_and_plot_top10_snp_overall(model, dataset, device, disease_name, disease_output_dir, config):
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    snp_importance = np.zeros(dataset.snp_dim)
    total_samples = 0

    for batch in data_loader:
        snp_feat, blood_feat, y = batch
        snp_feat = snp_feat.to(device).requires_grad_(True)
        blood_feat = blood_feat.to(device)
        y = y.to(device)

        for param in model.parameters():
            param.requires_grad = False

        outputs = model(snp_feat, blood_feat)
        loss = nn.CrossEntropyLoss()(outputs, y)

        if snp_feat.grad is not None:
            snp_feat.grad.zero_()

        loss.backward()

        if snp_feat.grad is not None:
            snp_importance += snp_feat.grad.abs().mean(dim=0).cpu().detach().numpy()

        total_samples += len(y)

    snp_importance = snp_importance / total_samples
    print(f"Using {total_samples} samples to calculate SNP importance")

    total_importance = np.sum(snp_importance)
    if total_importance == 0:
        print("Warning: All SNP importance values are zero, cannot plot pie chart")
        return None

    top10_indices = np.argsort(snp_importance)[::-1][:config.top_k]
    top10_names = [dataset.valid_snp_names[idx] for idx in top10_indices]
    top10_importance = snp_importance[top10_indices]
    other_importance = total_importance - np.sum(top10_importance)

    labels = top10_names + ["All Other SNPs"]
    sizes = list(top10_importance) + [other_importance]
    percentages = [(size / total_importance) * 100 for size in sizes]

    # text
    snp_pie_filename = f"{disease_name}_top10_snp_vs_others.pdf"
    full_save_path = os.path.join(disease_output_dir, snp_pie_filename)

    plt.figure(figsize=(14, 12), dpi=config.dpi)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, config.top_k))
    colors = list(colors) + [(0.8, 0.8, 0.8, 0.7)]

    # text
    max_explode = 0.15
    min_explode = 0.05
    explode = [max_explode - (i * (max_explode - min_explode) / (config.top_k - 1)) 
               for i in range(config.top_k)] + [0.08]

    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 1.5 else '',
        startangle=140,
        pctdistance=0.82,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
        textprops=dict(zorder=3),
        shadow=True,
        counterclock=False,
        normalize=True
    )

    # text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
        autotext.set_bbox(dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="black", alpha=0.7))

    # text
    for i, text in enumerate(texts):
        if i == len(texts) - 1:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        else:
            text.set_fontsize(10 if percentages[i] < 5 else 11)
        text.set_color('#2d3436')
        text.set_fontweight('medium')
        pos = text.get_position()
        text.set_position((pos[0] * 1.15, pos[1] * 1.15))

    # text
    centre_circle = plt.Circle((0, 0), 0.45, color='white', fc='white', edgecolor='#e0e0e0', linewidth=2)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # text
    plt.text(0, 0, f'TOP {config.top_k} SNPs\nvs Others', 
             ha='center', va='center', fontsize=16, fontweight='bold', color='#2d3436', linespacing=1.5)

    # text
    plt.title(f'TOP {config.top_k} SNPs vs All Other SNPs\nImportance Distribution for {disease_name}', 
             fontsize=20, fontweight='bold', pad=30, color='#2d3436')
    plt.figtext(0.5, 0.01, 
               'Percentages represent the proportion of each segment relative to the total importance of all SNPs',
               ha='center', fontsize=11, color='#636e72', style='italic')

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(full_save_path, format='pdf', dpi=config.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"TOP10 SNP vs others pie chart saved to: {full_save_path}")

    result_df = pd.DataFrame({
        'SNP Name': labels,
        'Importance': sizes,
        'Percentage(%)': [round(p, 2) for p in percentages]
    })
    return result_df


# ---------------------- SHAPtext ----------------------
def explain_with_shap(model, dataloader, feature_names, modal_name, save_path, config, feature_dim):
    """
    text：textdataloadertext，text
    feature_dim: text（dataset.snp_dim text dataset.blood_dim）
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    
    # 1. text
    modal_to_idx = {"SNP": 0, "text": 1}
    if modal_name not in modal_to_idx:
        raise ValueError(f"text：{modal_name}，text {list(modal_to_idx.keys())}")
    target_modal_idx = modal_to_idx[modal_name]
    
    # 2. textdataloadertext（text：text，text）
    target_modal_data = []  # text (n_samples, feature_dim)
    other_modal_data = []   # text (n_samples, other_feature_dim)
    
    for batch in dataloader:
        snp_batch, blood_batch, _ = batch  # text，text
        # text
        if target_modal_idx == 0:  # textSNP
            target_batch = snp_batch.cpu().numpy()
            other_batch = blood_batch.cpu().numpy()
        else:  # text
            target_batch = blood_batch.cpu().numpy()
            other_batch = snp_batch.cpu().numpy()
        
        target_modal_data.append(target_batch)
        other_modal_data.append(other_batch)
    
    # text
    target_modal_data = np.concatenate(target_modal_data, axis=0)  # (total_val_samples, feature_dim)
    other_modal_data = np.concatenate(other_modal_data, axis=0)    # (total_val_samples, other_feature_dim)
    
    # text（text）
    total_val_samples = target_modal_data.shape[0]
    target_sample_size = min(config.shap_sample_size, total_val_samples)
    target_data = target_modal_data[:target_sample_size]  # (target_sample_size, feature_dim)
    other_data = other_modal_data[:target_sample_size]    # (target_sample_size, other_feature_dim)
    
    print(f"⚠️ {modal_name} SHAPtext：text{total_val_samples}，text{target_sample_size}text，text{feature_dim}")
    print(f"✅ text：{target_data.shape}（text×text），text{target_data.size}")
    
    # text：textfeature_dimtext
    if target_data.shape[1] != feature_dim:
        raise ValueError(
            f"{modal_name}text！text{target_data.shape[1]}text，text{feature_dim}text\n"
            "text（textX_snp/X_bloodtext(n_samples, feature_dim)）"
        )
    
    # 3. texttorchtext（text）
    target_tensor = torch.FloatTensor(target_data).to(device)  # (target_sample_size, feature_dim)
    other_tensor = torch.FloatTensor(other_data).to(device)    # (target_sample_size, other_feature_dim)
    
    # 4. text（text，text）
    class SingleModalWrapper(nn.Module):
        def __init__(self, base_model, other_tensor, target_modal_idx):
            super().__init__()
            self.base_model = base_model
            self.other_tensor = other_tensor  # text
            self.target_modal_idx = target_modal_idx
        
        def forward(self, x):
            """x: text (batch_size, feature_dim)"""
            x = x.requires_grad_(True)  # text
            batch_size = x.shape[0]
            
            # textbatch_sizetext
            other_batch = self.other_tensor[:batch_size]
            
            # text（SNP, blood）
            if self.target_modal_idx == 0:
                model_input = (x, other_batch)
            else:
                model_input = (other_batch, x)
            
            # text
            logits = self.base_model(*model_input)
            probs = torch.softmax(logits, dim=1)
            return probs
    
    # 5. textSHAPtext
    wrapper_model = SingleModalWrapper(
        base_model=model,
        other_tensor=other_tensor,
        target_modal_idx=target_modal_idx
    ).to(device)
    wrapper_model.eval()
    
    # 6. textSHAPtext（text）
    explainer = shap.GradientExplainer(
        model=wrapper_model,
        data=target_tensor[:min(100, target_sample_size)],  # text100text
        local_smoothing=0.1
    )
    
    # textSHAPtext（text）
    shap_values = explainer.shap_values(target_tensor)
    
    # 7. textSHAPtext（text：textSHAPtext）
    num_classes = len(np.unique(dataloader.dataset.dataset.y))  # text
    if num_classes == 2:
        # text：text（text1）textSHAPtext
        shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
    else:
        # text：textSHAPtext（text）
        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values[:, :, 0]
    
    # text（text [text, text]）
    shap_values = np.squeeze(shap_values)

    # textSHAPtext
    if shap_values.shape != target_data.shape:
        raise ValueError(
            f"SHAPtext！text({target_sample_size}, {feature_dim})，text{shap_values.shape}\n"
            f"text: {num_classes}，text"
        )

    
    # 8. text
    valid_feat_num = min(len(feature_names), feature_dim)
    plot_feature_names = feature_names[:valid_feat_num]
    shap_vals_for_plot = shap_values[:, :valid_feat_num]
    target_data_for_plot = target_data[:, :valid_feat_num]
    
    # text（SHAPtext）
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
        
        plt.grid(False)  # text
        plt.xlabel('SHAPtext（text，text）', fontsize=14, fontweight='bold')
        plt.title(f'{modal_name}textSHAPtextTOP{config.top_k}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{modal_name}_shap_sum_bar.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✅ {modal_name} SHAPtext")
    
    # text
    if config.shap_plot_type in ["both", "beeswarm"]:
        plt.figure(figsize=(12, 8), dpi=config.dpi)
        shap.summary_plot(
            shap_vals_for_plot,
            target_data_for_plot,
            feature_names=plot_feature_names,
            show=False,
            max_display=config.top_k
        )
        plt.grid(False)  # text
        plt.title(f'{modal_name}textSHAPtext', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('SHAPtext', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{modal_name}_shap_beeswarm.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✅ {modal_name} SHAPtext")
    
    # 9. textSHAPtext
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
    
    print(f"✅ {modal_name} SHAPtext")
    return shap_values


def plot_mean_shap_bar(mean_shap_df, modal_name, save_path, top_k=10):
    """text5textSHAPtext"""
    # textSHAPtext
    feature_cols = mean_shap_df.columns
    # text（text）
    feature_names = [col.split(f"{modal_name}_")[-1] for col in feature_cols]
    shap_means = mean_shap_df.values[0]  # text
    
    # text，textTOP-Ktext
    sorted_indices = np.argsort(np.abs(shap_means))[-top_k:][::-1]  # text
    top_names = [feature_names[i] for i in sorted_indices]
    top_means = shap_means[sorted_indices]
    
    # text
    plt.figure(figsize=(12, 8), dpi=150)
    colors = ['#E63946' if val > 0 else '#2E86AB' for val in top_means]  # text=text，text=text  
    bars = plt.barh(top_names, top_means, color=colors)
    
    # text
    for bar in bars:
        width = bar.get_width()
        # text（text）
        text_x = width + 0.0005 if width > 0 else width - 0.0005
        plt.text(
            text_x, 
            bar.get_y() + bar.get_height()/2,
            f"{width:.6f}",  # text6text
            va='center', 
            fontsize=10
        )
    
    # text
    plt.xlabel(f'5textSHAPtext（text，text）', fontsize=14, fontweight='bold')
    plt.title(f'{modal_name}text - 5textSHAPtextTOP{top_k}', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()  # text（text）
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # text
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ text{modal_name}textSHAPtext：{save_path}")


# ---------------------- Performance Metrics Scatter Plot ----------------------
def plot_performance_metrics_scatter(disease_name, disease_output_dir, results, config):
    metrics_scatter_filename = f"{disease_name}_performance_metrics_scatter.pdf"
    full_save_path = os.path.join(disease_output_dir, metrics_scatter_filename)

    labels = [f'Fold {i+1}' for i in range(len(results))]
    
    metrics_data = {}
    for i, label in enumerate(labels):
        metrics_data[label] = [
            results[i]['precision'],
            results[i]['recall'],
            results[i]['f1'],
            results[i]['auc'] if results[i]['auc'] is not None else 0
        ]
    
    metrics = ['Precision', 'Recall', 'F1 Score', 'AUC Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(12, 8), dpi=config.dpi)
    x_pos = np.arange(len(metrics))
    width = 0.15

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
    
    all_values = [
        [res['precision'], res['recall'], res['f1'], res['auc'] if res['auc'] is not None else 0]
        for res in results
    ]
    mean_values = np.mean(all_values, axis=0)
    plt.plot(
        x_pos + width * (len(labels) - 1) / 2,
        mean_values,
        color='red',
        linestyle='--',
        linewidth=2,
        marker='o',
        markersize=8,
        label='Mean'
    )

    plt.xticks(x_pos + width * (len(labels) - 1) / 2, metrics, fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=config.n_splits + 1, fontsize=10, frameon=True)

    plt.title(f'Performance Metrics Across Cross-Validation Folds\nfor {disease_name}', 
             fontsize=16, fontweight='bold', pad=30)
    plt.ylabel('Score Value', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(full_save_path, format='pdf', dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"Performance metrics scatter plot saved to: {full_save_path}")

    return {
        'metrics': metrics,
        'mean': mean_values,
        'std': np.std(all_values, axis=0)
    }


# ---------------------- Save Metrics to CSV ----------------------
def save_metrics_to_csv(disease_name, disease_output_dir, metrics_summary, config):
    metrics_csv_filename = f"{disease_name}_performance_metrics_summary.csv"
    full_save_path = os.path.join(disease_output_dir, metrics_csv_filename)
    
    df = pd.DataFrame({
        'Metric': metrics_summary['metrics'],
        'Mean': [round(v, 4) for v in metrics_summary['mean']],
        'Standard Deviation': [round(v, 4) for v in metrics_summary['std']]
    })
    
    df.to_csv(full_save_path, index=False, encoding='utf-8-sig')
    print(f"Performance metrics summary saved to CSV: {full_save_path}")


# ---------------------- Mean ROC Curve Plot ----------------------
def plot_mean_roc_curve(all_true_labels, all_pred_probs, all_aucs, save_dir, save_filename):
    """textROCtext"""
    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, save_filename)
    print(f"\nMean ROC curve will be saved to: {full_save_path}")
    
    # text
    valid_folds = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (y_true, y_score, roc_auc) in enumerate(zip(all_true_labels, all_pred_probs, all_aucs)):
        if len(np.unique(y_true)) != 2 or roc_auc is None:
            continue
        valid_folds.append(i)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        f = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
        tprs.append(f(mean_fpr))
    
    if not valid_folds:
        print("❌ No valid binary classification data, cannot plot mean ROC curve")
        return
    
    # text
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean([all_aucs[i] for i in valid_folds])
    std_auc = np.std([all_aucs[i] for i in valid_folds])
    std_tpr = np.std(tprs, axis=0)
    
    # textROCtext
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='#2E86AB', lw=4, alpha=0.9,
             label=f'Mean ROC Curve (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#A23B72', alpha=0.2,
                     label='±1 Standard Deviation')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='#F18F01', alpha=0.8,
             label='Random Guess (AUC = 0.5)')
    
    # text
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    plt.title(f'5-Fold Cross-Validation - Mean ROC Curve\nfor {save_filename.split("_")[0]}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    # text
    plt.savefig(full_save_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Mean ROC curve saved: {full_save_path}")


# ---------------------- Loss Curve Plot ----------------------
def plot_loss_curves(all_train_losses, all_val_losses, disease_name, disease_output_dir, config):
    loss_filename = f"{disease_name}_loss_curves_with_mean.pdf"
    full_save_path = os.path.join(disease_output_dir, loss_filename)
    
    plt.figure(figsize=(10, 6), dpi=config.dpi)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # text
    for i, (color, train_losses, val_losses) in enumerate(zip(
        colors, all_train_losses, all_val_losses
    )):
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, color=color, linestyle='--', lw=1.5, alpha=0.4,
                 label=f'Fold {i + 1} - Training' if i == 0 else "")
        plt.plot(epochs, val_losses, color=color, lw=1.5, alpha=0.4,
                 label=f'Fold {i + 1} - Validation' if i == 0 else "")
    
    # text
    mean_train_losses = np.mean(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    epochs = range(1, len(mean_train_losses) + 1)
    plt.plot(epochs, mean_train_losses, color='navy', lw=3, alpha=0.9,
             label='Mean Training Loss', marker='o', markersize=5, markevery=10)
    plt.plot(epochs, mean_val_losses, color='darkorange', lw=3, alpha=0.9,
             label='Mean Validation Loss', marker='s', markersize=5, markevery=10)
    
    # text
    plt.xlim([1, len(epochs)])
    plt.ylim([0, max(np.max(all_train_losses), np.max(all_val_losses)) * 1.1])
    plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
    plt.title(f'Training and Validation Loss Curves\nfor {disease_name}', 
             fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="upper right", fontsize=12, frameon=True)
    plt.grid(alpha=0.3)
    
    # text
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig(full_save_path, format='pdf', dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curves saved: {full_save_path}")


# ---------------------- Main Execution ----------------------
def main():
    # text
    config = Config()
    
    # text
    if not os.path.exists(config.root_data_dir):
        print(f"Error: text - {config.root_data_dir}")
        return
    
    # textbloodtext
    if not os.path.exists(config.blood_path):
        print(f"Error: text - {config.blood_path}")
        return
    
    # text
    disease_folders = [f for f in os.listdir(config.root_data_dir) 
                       if os.path.isdir(os.path.join(config.root_data_dir, f))]
    
    if not disease_folders:
        print(f"text: {config.root_data_dir}")
        return
    
    # text
    for disease_name in disease_folders:
        print(f"\n{'='*50}")
        print(f"text: {disease_name}")
        print(f"{'='*50}")
        
        # text
        snp_path = os.path.join(config.root_data_dir, disease_name, f"{disease_name}.csv")          
        label_path = os.path.join(config.root_data_dir, disease_name, f"{disease_name}_label.csv")   
        
        if not os.path.exists(snp_path):
            print(f"text: SNPtext - {snp_path}，text")
            continue
        
        if not os.path.exists(label_path):
            print(f"text: text - {label_path}，text")
            continue
        
        # 1. text
        try:
            dataset = MultiModalDataset(config, disease_name)
        except Exception as e:
            print(f"text: {str(e)}，text")
            continue
        
        print(f"text: {len(dataset)} text，text: {np.bincount(dataset.y)}")
        
        # 2. text
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"text: {device} (GPUtext: {torch.cuda.is_available()})")
        
        # 3. text
        results = []
        all_train_losses = []
        all_val_losses = []
        all_true_labels = []
        all_pred_probs = []
        all_aucs = []
        
        # textSHAPtext
        snp_shap_sums = []
        blood_shap_sums = []
        
        # text（text）
        best_model = None
        best_f1 = 0.0
        
        # 4. text
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.X[0], dataset.y), 1):
            print(f"\n===== text {fold} text =====")
            
            # text
            train_ds = DatasetFromIndices(dataset, train_idx)
            val_ds = DatasetFromIndices(dataset, val_idx)
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            
            # text（text）
            model = MultiModalTransformerModel(
                snp_dim=dataset.snp_dim,
                blood_dim=dataset.blood_dim,
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                num_encoder_layers=config.num_encoder_layers,
                num_classes=len(np.unique(dataset.y))
            ).to(device)
            
            # text
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # text
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
            
            # text
            train_losses = []
            val_losses = []
            
            # text
            current_best_f1 = 0.0
            current_best_model = None
            for epoch in range(1, config.epochs + 1):
                # text
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                
                # text
                val_metrics = evaluate(model, val_loader, criterion, device)
                
                # text
                train_losses.append(train_loss)
                val_losses.append(val_metrics['val_loss'])
                
                # text
                auc_log = f"AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] is not None else "AUC: N/A"
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                      f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | "
                      f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f} | {auc_log}")
                
                # text（textF1text）
                if val_metrics['f1'] > current_best_f1:
                    current_best_f1 = val_metrics['f1']
                    current_best_model = model.state_dict()  # text
                    print(f"★ text{fold}textF1: {current_best_f1:.4f}")
                
                # text
                scheduler.step(val_metrics['f1'])
            
            # text
            model.load_state_dict(current_best_model)
            
            # text（textF1text）
            if current_best_f1 > best_f1:
                best_f1 = current_best_f1
                best_model = model  # text
            
            # text
            results.append({
                "fold": fold, "accuracy": val_metrics['acc'], "precision": val_metrics['precision'],
                "recall": val_metrics['recall'], "f1": current_best_f1, "auc": val_metrics['auc']
            })
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_true_labels.append(val_metrics['true_labels'])
            all_pred_probs.append(val_metrics['pred_probs'])
            all_aucs.append(val_metrics['auc'])
            
            # textSHAPtext（text）
            disease_output_dir = os.path.join(config.root_data_dir, disease_name, config.output_subdir)
            os.makedirs(disease_output_dir, exist_ok=True)
            fold_shap_dir = os.path.join(disease_output_dir, f"fold_{fold}")
            os.makedirs(fold_shap_dir, exist_ok=True)
            
            # textSNPtextSHAPtext（textdataset.snp_dim）
            try:
                snp_shap = explain_with_shap(
                    model=model,
                    dataloader=val_loader,
                    feature_names=dataset.valid_snp_names,
                    modal_name="SNP",
                    save_path=fold_shap_dir,
                    config=config,
                    feature_dim=dataset.snp_dim  # text：textSNPtext
                )
                snp_shap_sums.append(np.sum(snp_shap, axis=0))
            except Exception as e:
                print(f"textSNPtextSHAPtext: {str(e)}")
            
            # textSHAPtext（textdataset.blood_dim）
            try:
                blood_shap = explain_with_shap(
                    model=model,
                    dataloader=val_loader,
                    feature_names=dataset.valid_blood_names,
                    modal_name="text",
                    save_path=fold_shap_dir,
                    config=config,
                    feature_dim=dataset.blood_dim  # text：text
                )
                blood_shap_sums.append(np.sum(blood_shap, axis=0))
            except Exception as e:
                print(f"textSHAPtext: {str(e)}")
        
        # 5. text（text）
        print("\n===== text =====")
        try:
            # text
            risk_df = predict_sample_risk(
                model=best_model,
                dataset=dataset,
                device=device,
                config=config
            )
            # text
            plot_risk_distribution(
                risk_df=risk_df,
                disease_name=disease_name,
                save_dir=disease_output_dir,
                config=config
            )
        except Exception as e:
            print(f"text: {str(e)}")
        
        # text5textSHAPtext
        if snp_shap_sums:
            snp_mean_shap = np.mean(snp_shap_sums, axis=0)
            snp_mean_df = pd.DataFrame([snp_mean_shap], columns=[f"SNP_{name}" for name in dataset.valid_snp_names])
            plot_mean_shap_bar(
                snp_mean_df, 
                "SNP", 
                os.path.join(disease_output_dir, f"{disease_name}_snp_mean_shap_bar.pdf"),
                top_k=config.top_k
            )
        
        if blood_shap_sums:
            blood_mean_shap = np.mean(blood_shap_sums, axis=0)
            blood_mean_df = pd.DataFrame([blood_mean_shap], columns=[f"text_{name}" for name in dataset.valid_blood_names])
            plot_mean_shap_bar(
                blood_mean_df, 
                "text", 
                os.path.join(disease_output_dir, f"{disease_name}_blood_mean_shap_bar.pdf"),
                top_k=config.top_k
            )
        
        # 6. text
        print("\n===== 5text =====")
        for res in results:
            auc_str = f"{res['auc']:.4f}" if res['auc'] is not None else "N/A"
            print(
                f"text {res['fold']} | text: {res['accuracy']:.4f} | "
                f"text: {res['precision']:.4f} | text: {res['recall']:.4f} | "
                f"F1text: {res['f1']:.4f} | AUC: {auc_str}"
            )
        
        # 7. text
        print("\n===== text =====")
        try:
            metrics_summary = plot_performance_metrics_scatter(
                disease_name=disease_name,
                disease_output_dir=disease_output_dir,
                results=results,
                config=config
            )
        except Exception as e:
            print(f"text: {str(e)}")
            metrics_summary = None
        
        # 8. textCSV
        if metrics_summary:
            print("\n===== textCSV =====")
            try:
                save_metrics_to_csv(
                    disease_name=disease_name,
                    disease_output_dir=disease_output_dir,
                    metrics_summary=metrics_summary,
                    config=config
                )
            except Exception as e:
                print(f"textCSVtext: {str(e)}")
        
                # 9. textROCtext
        valid_aucs = [r['auc'] for r in results if r['auc'] is not None]
        if valid_aucs:
            print("\n===== textROCtext =====")
            try:
                valid_true_labels = [tl for tl, auc in zip(all_true_labels, all_aucs) if auc is not None]
                valid_pred_probs = [pp for pp, auc in zip(all_pred_probs, all_aucs) if auc is not None]
                plot_mean_roc_curve(
                    valid_true_labels, 
                    valid_pred_probs, 
                    valid_aucs,
                    save_dir=disease_output_dir,
                    save_filename=f"{disease_name}_mean_roc_curve.pdf"
                )
            except Exception as e:
                print(f"textROCtext: {str(e)}")
        else:
            print("❌ text，textROCtext")
        
        # 10. text
        print("\n===== text =====")
        try:
            plot_loss_curves(
                all_train_losses=all_train_losses,
                all_val_losses=all_val_losses,
                disease_name=disease_name,
                disease_output_dir=disease_output_dir,
                config=config
            )
        except Exception as e:
            print(f"text: {str(e)}")
        
        # 11. textTOP10 SNP vs textSNPtext
        print("\n===== textTOP10 SNPstext =====")
        try:
            snp_result_df = calculate_and_plot_top10_snp_overall(
                model=best_model,  # text
                dataset=dataset,
                device=device,
                disease_name=disease_name,
                disease_output_dir=disease_output_dir,
                config=config
            )
            if snp_result_df is not None:
                print("\nTOP10 SNPs vs text SNPs text:")
                print(snp_result_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"textTOP10 SNPtext: {str(e)}")
        
        print(f"\n{'='*50}")
        print(f"text {disease_name} text")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()