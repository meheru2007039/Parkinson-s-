# ============================================================================
# Imports 
# ============================================================================
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pathlib
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from scipy.signal import butter, filtfilt
import random
import os
import json
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

# ============================================================================
# config
# ============================================================================
def get_config():
    config = {
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'apply_prepare_text': False,
        'split_type': 3,
        'split_ratio': 0.85,
        'train_tasks': None,
        'num_folds': 5,

        'input_dim': 6,
        'model_dim': 64,
        'num_heads': 8,
        'num_layers': 3,
        'num_window_layers': 4,
        'num_task_layers': 2,
        'd_ff': 256,
        'dropout': 0.2,
        'seq_len': 256,
        'num_classes': 2,
        'use_text': False,

        'batch_size': 8,  # Actual batch size in memory
        'gradient_accumulation_steps': 8,  # Effective batch size = 8 * 8 = 64
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,
        'max_grad_norm': 1.0,  # Gradient clipping threshold (0 = no clipping)

        'save_metrics': True,
        'create_plots': True,
    }
    
    return config
# ============================================================================
# Helper functions 
# ============================================================================
def create_windows(data, window_size=256, overlap=0):
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))   
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])
    
    return np.array(windows) if windows else None


def downsample(data, original_freq=100, target_freq=64):
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
    nyquist = 0.5 * original_freq
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ============================================================================
# DataLoader
# ============================================================================
class ParkinsonsDatasetLoader(Dataset):
    def __init__(self, data_root: str = None, window_size: int = 256,
                 max_windows_per_task: int = 10,  # Max windows to include per task
                 min_windows_per_task: int = 2,   # Min windows required for a task
                 patient_task_data=None,          # For split datasets
                 apply_downsampling=True,
                 apply_bandpass_filter=True):

        self.window_size = window_size
        self.max_windows_per_task = max_windows_per_task
        self.min_windows_per_task = min_windows_per_task
        self.apply_downsampling = apply_downsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.data_root = data_root

        # Store patient-task combinations
        # Each entry: {patient_id, task_name, left_windows, right_windows,
        #              hc_vs_pd, pd_vs_dd, num_windows}
        self.patient_task_data = []
        
        if data_root is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"

            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            
            self.patient_ids_list = list(range(1, 470))
            print(f"Loading dataset: {len(self.patient_ids_list)} patients")
            
            self._load_data()
        elif patient_task_data is not None:
            # For split datasets
            self.patient_task_data = patient_task_data

        print(f"Total patient-task combinations: {len(self.patient_task_data)}")
    
    
    def _load_data(self):
        """Load data organized by patient-task combinations"""
        
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patient-task data"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))

            if not patient_path.exists():
                continue

            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)

                condition = metadata.get('condition', '')

                # Determine labels
                # UNIFORM overlap to prevent num_windows leakage
                # Different overlaps create different window counts which leak labels!
                overlap = 0.5

                if condition == 'Healthy':
                    hc_vs_pd_label = 0
                    pd_vs_dd_label = -1
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1
                    pd_vs_dd_label = 0
                else:
                    hc_vs_pd_label = -1
                    pd_vs_dd_label = 1
                
                # Process each task separately
                for task in self.tasks:
                    left_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="LeftWrist"))
                    right_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="RightWrist"))
                    
                    if not (left_path.exists() and right_path.exists()):
                        continue
                    
                    try:
                        # Load and preprocess data
                        left_data = np.loadtxt(left_path, delimiter=",")
                        right_data = np.loadtxt(right_path, delimiter=",")
                        
                        # Take first 6 channels and skip first 0.5 sec
                        if left_data.shape[1] > 6:
                            left_data = left_data[:, :6]
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :]
                        
                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]
                        
                        # Downsample
                        if self.apply_downsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)
                        
                        # Bandpass filter
                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)
                        
                        if left_data is None or right_data is None:
                            continue
                        
                        # Create windows
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)
                        
                        if left_windows is not None and right_windows is not None:
                            min_windows = min(len(left_windows), len(right_windows))
                            
                            if min_windows >= self.min_windows_per_task:
                                # Limit to max_windows_per_task
                                num_windows = min(min_windows, self.max_windows_per_task)
                                
                                # Store this patient-task combination
                                self.patient_task_data.append({
                                    'patient_id': patient_id,
                                    'task_name': task,
                                    'left_windows': left_windows[:num_windows],  # shape: (num_windows, 256, 6)
                                    'right_windows': right_windows[:num_windows], # shape: (num_windows, 256, 6)
                                    'hc_vs_pd': hc_vs_pd_label,
                                    'pd_vs_dd': pd_vs_dd_label,
                                    'num_windows': num_windows
                                })
                    
                    except Exception as e:
                        print(f"Error loading patient {patient_id}, task {task}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
    
    
    def get_k_fold_split(self, k=5):
        """
        K-fold split at PATIENT level to prevent data leakage.
        Returns list of (train_dataset, test_dataset) tuples.
        """
        if self.data_root is None:
            raise ValueError("data_root is required for K-fold split")
        
        # Get unique patients and their conditions
        patient_conditions = {}
        patients_template = pathlib.Path(self.data_root) / "patients" / "patient_{p:03d}.json"
        
        for patient_id in range(1, 470):
            patient_path = pathlib.Path(str(patients_template).format(p=patient_id))
            if patient_path.exists():
                try:
                    with open(patient_path, 'r') as f:
                        condition = json.load(f).get('condition', 'Unknown')
                        patient_conditions[patient_id] = condition
                except:
                    pass
        
        # Create patient list with labels
        patient_list = []
        patient_labels = []
        for pid in sorted(patient_conditions.keys()):
            condition = patient_conditions[pid]
            if condition == 'Healthy':
                label = 0
            elif 'Parkinson' in condition:
                label = 1
            else:
                label = 2
            patient_list.append(pid)
            patient_labels.append(label)
        
        print(f"\nTotal patients: {len(patient_list)} (HC={patient_labels.count(0)}, PD={patient_labels.count(1)}, DD={patient_labels.count(2)})")
        
        # Stratified K-fold split
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_datasets = []
        
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(patient_list, patient_labels)):
            train_patients = set([patient_list[i] for i in train_idx])
            test_patients = set([patient_list[i] for i in test_idx])

            # ============ DATA LEAKAGE CHECK ============
            overlap = train_patients.intersection(test_patients)
            if overlap:
                print(f"\n⚠️  WARNING: FOLD {fold_id+1} HAS DATA LEAKAGE!")
                print(f"   Overlapping patients: {sorted(list(overlap))[:10]}... (showing first 10)")
            else:
                print(f"\n✓ Fold {fold_id+1}: No patient overlap detected")

            print(f"   Train patients: {len(train_patients)} (IDs: {sorted(list(train_patients))[:5]}...)")
            print(f"   Test patients:  {len(test_patients)} (IDs: {sorted(list(test_patients))[:5]}...)")
            # ============================================

            # Split patient-task data
            train_data = [pt for pt in self.patient_task_data if pt['patient_id'] in train_patients]
            test_data = [pt for pt in self.patient_task_data if pt['patient_id'] in test_patients]
            
            # Create datasets
            train_dataset = ParkinsonsDatasetLoader(
                data_root=None,
                patient_task_data=train_data,
                window_size=self.window_size,
                max_windows_per_task=self.max_windows_per_task,
                min_windows_per_task=self.min_windows_per_task
            )
            
            test_dataset = ParkinsonsDatasetLoader(
                data_root=None,
                patient_task_data=test_data,
                window_size=self.window_size,
                max_windows_per_task=self.max_windows_per_task,
                min_windows_per_task=self.min_windows_per_task
            )
            
            # Print fold statistics
            train_hc = sum(1 for pt in train_data if pt['hc_vs_pd'] == 0)
            train_pd = sum(1 for pt in train_data if pt['hc_vs_pd'] == 1 and pt['pd_vs_dd'] == 0)
            train_dd = sum(1 for pt in train_data if pt['pd_vs_dd'] == 1)
            test_hc = sum(1 for pt in test_data if pt['hc_vs_pd'] == 0)
            test_pd = sum(1 for pt in test_data if pt['hc_vs_pd'] == 1 and pt['pd_vs_dd'] == 0)
            test_dd = sum(1 for pt in test_data if pt['pd_vs_dd'] == 1)
            
            print(f"\nFold {fold_id+1}/{k}:")
            print(f"  Train: {len(train_data)} patient-task pairs (HC={train_hc}, PD={train_pd}, DD={train_dd})")
            print(f"  Test:  {len(test_data)} patient-task pairs (HC={test_hc}, PD={test_pd}, DD={test_dd})")
            
            fold_datasets.append((train_dataset, test_dataset))
        
        return fold_datasets
    
    
    def __len__(self):
        return len(self.patient_task_data)
    
    
    def __getitem__(self, idx):
        """Returns ONE patient-task combination with ALL its windows."""
        pt_data = self.patient_task_data[idx]

        left_windows = torch.FloatTensor(pt_data['left_windows'])   # (num_windows, 256, 6)
        right_windows = torch.FloatTensor(pt_data['right_windows']) # (num_windows, 256, 6)
        hc_vs_pd = torch.LongTensor([pt_data['hc_vs_pd']])
        pd_vs_dd = torch.LongTensor([pt_data['pd_vs_dd']])
        num_windows = pt_data['num_windows']
        patient_id = pt_data['patient_id']
        task_name = pt_data['task_name']

        return {
            'left_windows': left_windows,
            'right_windows': right_windows,
            'hc_vs_pd': hc_vs_pd.squeeze(),
            'pd_vs_dd': pd_vs_dd.squeeze(),
            'num_windows': num_windows,
            'patient_id': patient_id,
            'task_name': task_name
        }


# ============================================================================
# Custom collate function for variable-length sequences
# ============================================================================
def collate_fn(batch):
    """
    Collate function to handle variable number of windows per task.
    Pads to max_windows in the batch.
    """
    # Find max windows in this batch
    max_windows = max([item['num_windows'] for item in batch])
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    left_windows_padded = torch.zeros(batch_size, max_windows, 256, 6)
    right_windows_padded = torch.zeros(batch_size, max_windows, 256, 6)
    masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)  # True for valid windows
    
    hc_vs_pd_labels = []
    pd_vs_dd_labels = []
    num_windows_list = []
    patient_ids = []
    task_names = []

    for i, item in enumerate(batch):
        num_win = item['num_windows']

        # Copy actual data
        left_windows_padded[i, :num_win] = item['left_windows']
        right_windows_padded[i, :num_win] = item['right_windows']
        masks[i, :num_win] = True

        hc_vs_pd_labels.append(item['hc_vs_pd'])
        pd_vs_dd_labels.append(item['pd_vs_dd'])
        num_windows_list.append(num_win)
        patient_ids.append(item['patient_id'])
        task_names.append(item['task_name'])

    return {
        'left_windows': left_windows_padded,
        'right_windows': right_windows_padded,
        'masks': masks,
        'hc_vs_pd': torch.stack(hc_vs_pd_labels),
        'pd_vs_dd': torch.stack(pd_vs_dd_labels),
        'num_windows': num_windows_list,
        'patient_ids': patient_ids,
        'task_names': task_names
    }

# ============================================================================
# model
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.cross_attention_2to1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        self.self_attention_1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        
        # Layer norms for residual connections
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        # Cross attention with residual connections
        channel_1_cross_attn, _ = self.cross_attention_1to2(query=channel_1,key=channel_2,value=channel_2)
        channel_1_cross = self.norm_cross_1(channel_1 + channel_1_cross_attn)
        
        channel_2_cross_attn, _ = self.cross_attention_2to1(query=channel_2,key=channel_1,value=channel_1)
        channel_2_cross = self.norm_cross_2(channel_2 + channel_2_cross_attn)
        
        # Self attention with residual connections
        channel_1_self_attn, _ = self.self_attention_1(query=channel_1_cross,key=channel_1_cross,value=channel_1_cross)
        channel_1_self = self.norm_self_1(channel_1_cross + channel_1_self_attn)
        
        channel_2_self_attn, _ = self.self_attention_2(query=channel_2_cross,key=channel_2_cross,value=channel_2_cross)
        channel_2_self = self.norm_self_2(channel_2_cross + channel_2_self_attn)
        
        # Feed forward
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out

class MyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_window_layers: int = 4,  # Number of cross-attention layers at window level
        num_task_layers: int = 2,    # Number of attention layers at task level
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 256,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.seq_len = seq_len

        # ========== LEVEL 1: Window-Level Processing ==========
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, max_len=seq_len)

        # Cross-attention layers
        self.window_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(num_window_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ========== LEVEL 2: Task-Level Processing ==========
        self.window_positional_encoding = PositionalEncoding(model_dim * 2, max_len=100)  # max 100 windows

        self.task_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=model_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(model_dim * 2),
                'feed_forward': FeedForward(model_dim * 2, d_ff, dropout)
            })
            for _ in range(num_task_layers)
        ])

        # Task-level pooling
        self.task_attention_pooling = nn.Sequential(
            nn.Linear(model_dim * 2, 1),
            nn.Softmax(dim=1)
        )

        # ========== Classification Heads ==========
        # No text encoder - using only signal features
        fusion_dim = model_dim * 2

        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: HC vs PD
        )

        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: PD vs DD
        )

        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, batch):
       
        device = batch['left_windows'].device
        batch_size = batch['left_windows'].shape[0]
        max_windows = batch['left_windows'].shape[1]
        
        # ========== LEVEL 1: Window-Level Cross-Attention ==========
        # Reshape: (batch, max_windows, 256, 6) -> (batch * max_windows, 256, 6)
        left_windows_flat = batch['left_windows'].view(-1, self.seq_len, 6)
        right_windows_flat = batch['right_windows'].view(-1, self.seq_len, 6)
        
        # Project to model dimension
        left_encoded = self.left_projection(left_windows_flat)   # (batch*max_windows, 256, model_dim)
        right_encoded = self.right_projection(right_windows_flat) 
        
        # Add positional encoding
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        # Apply cross-attention layers between left and right wrist
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)
        
        # Global pooling for each window
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)  # (batch*max_windows, model_dim)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1) # (batch*max_windows, model_dim)
        
        # Concatenate left and right features
        window_features = torch.cat([left_pool, right_pool], dim=1)  # (batch*max_windows, model_dim*2)
        
        # ========== LEVEL 2: Task-Level Attention ==========
        # Reshape back to (batch, max_windows, model_dim*2)
        window_features = window_features.view(batch_size, max_windows, -1)
        
        window_features = self.window_positional_encoding(window_features)
        
        # Create attention mask for padding (invert the mask: True -> False for valid positions)
        key_padding_mask = ~batch['masks']  # (batch_size, max_windows)
        
        # Apply task-level self-attention layers
        task_features = window_features
        for task_layer in self.task_layers:
            attn_output, _ = task_layer['self_attention'](
                query=task_features,
                key=task_features,
                value=task_features,
                key_padding_mask=key_padding_mask
            )
            task_features = task_layer['norm'](task_features + attn_output)
            task_features = task_layer['feed_forward'](task_features)
        
        # Attention-based pooling to get task representation
        attention_weights = self.task_attention_pooling(task_features)  # (batch, max_windows, 1)
        
        attention_weights = attention_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        task_representation = (task_features * attention_weights).sum(dim=1)  # (batch, model_dim*2)

        # ========== Classification ==========
        # Using only signal features (no text)
        logits_hc_vs_pd = self.head_hc_vs_pd(task_representation)
        logits_pd_vs_dd = self.head_pd_vs_dd(task_representation)
        
        return logits_hc_vs_pd, logits_pd_vs_dd
    
    
    def get_features(self, batch):
        """Extract features for tsne plot"""
        device = batch['left_windows'].device
        batch_size = batch['left_windows'].shape[0]
        max_windows = batch['left_windows'].shape[1]
        
        # Level 1: Window processing
        left_windows_flat = batch['left_windows'].view(-1, self.seq_len, 6)
        right_windows_flat = batch['right_windows'].view(-1, self.seq_len, 6)
        
        left_encoded = self.left_projection(left_windows_flat)
        right_encoded = self.right_projection(right_windows_flat)
        
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)
        
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)
        
        window_features = torch.cat([left_pool, right_pool], dim=1)
        window_features = window_features.view(batch_size, max_windows, -1)
        
        # Level 2: Task processing
        window_features = self.window_positional_encoding(window_features)
        key_padding_mask = ~batch['masks']
        
        task_features = window_features
        for task_layer in self.task_layers:
            attn_output, _ = task_layer['self_attention'](
                query=task_features,
                key=task_features,
                value=task_features,
                key_padding_mask=key_padding_mask
            )
            task_features = task_layer['norm'](task_features + attn_output)
            task_features = task_layer['feed_forward'](task_features)
        
        attention_weights = self.task_attention_pooling(task_features)
        attention_weights = attention_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        task_representation = (task_features * attention_weights).sum(dim=1)

        return {
            'window_features': window_features,  # (batch, max_windows, model_dim*2)
            'task_representation': task_representation,  # (batch, model_dim*2)
            'fused_features': task_representation,  # (batch, model_dim*2) - no text, just signal features
            'attention_weights': attention_weights.squeeze(-1)  # (batch, max_windows)
        }
# ============================================================================
# Evaluation functions
# ============================================================================
def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'confusion_matrix': cm
    }
    
    if verbose and task_name:
        print(f"\n=== {task_name} Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision_avg:.4f}")
        print(f" Recall: {recall_avg:.4f}")
        print(f"F1: {f1_avg:.4f}")
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_name = "HC" if label == 0 else ("PD" if label == 1 else f"Class_{label}")
                if task_name == "PD vs DD":
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics

def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd):

    os.makedirs("metrics", exist_ok=True)

    # helper writer
    def write_csv(filename, metrics_list):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
            for epoch_data in metrics_list:
                writer.writerow([
                    epoch_data['epoch'],
                    epoch_data['metrics'].get('accuracy', 0),
                    epoch_data['metrics'].get('precision', 0),
                    epoch_data['metrics'].get('recall', 0),
                    epoch_data['metrics'].get('f1', 0)
                ])

    # HC vs PD
    if fold_metrics_hc:
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")

    # PD vs DD
    if fold_metrics_pd:
        pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")



def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir="plots"):
    
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    valid_hc_pd = hc_pd_labels != -1
    valid_pd_dd = pd_dd_labels != -1

    # plot HC vs PD
    if np.any(valid_hc_pd):
        plt.figure(figsize=(8, 6))
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        pd_mask = labels_hc_pd == 1
        
        if np.any(hc_mask):
            plt.scatter(features_hc_pd[hc_mask,0], features_hc_pd[hc_mask,1], 
                        c='blue', label=f'HC (n={np.sum(hc_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask,0], features_hc_pd[pd_mask,1], 
                        c='red', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: HC vs PD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_hc_vs_pd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[saved] tsne_hc_vs_pd.png")

    # plot PD vs DD
    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]

        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask,0], features_pd_dd[pd_mask,1], 
                        c='green', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask,0], features_pd_dd[dd_mask,1], 
                        c='orange', label=f'DD (n={np.sum(dd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: PD vs DD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[saved] tsne_pd_vs_dd.png")

    return features_2d


# ============================================================================
# Gradient Monitoring Utilities
# ============================================================================
def check_gradients(model, log_prefix=""):
    """
    Check for gradient vanishing/exploding issues.
    Returns dict with gradient statistics.
    """
    total_norm = 0.0
    grad_stats = {
        'layer_norms': [],
        'layer_names': [],
        'max_grad': 0.0,
        'min_grad': float('inf'),
        'num_zero_grads': 0,
        'num_nan_grads': 0
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            grad_stats['layer_names'].append(name)
            grad_stats['layer_norms'].append(param_norm)
            grad_stats['max_grad'] = max(grad_stats['max_grad'], param_norm)
            grad_stats['min_grad'] = min(grad_stats['min_grad'], param_norm)

            if param_norm < 1e-7:
                grad_stats['num_zero_grads'] += 1
            if torch.isnan(param.grad).any():
                grad_stats['num_nan_grads'] += 1

    grad_stats['total_norm'] = total_norm ** 0.5

    # Check for issues
    if grad_stats['total_norm'] < 1e-5:
        print(f"{log_prefix}⚠️  WARNING: Very small gradient norm ({grad_stats['total_norm']:.2e}) - possible vanishing gradients!")
    elif grad_stats['total_norm'] > 100:
        print(f"{log_prefix}⚠️  WARNING: Very large gradient norm ({grad_stats['total_norm']:.2e}) - possible exploding gradients!")

    if grad_stats['num_zero_grads'] > len(grad_stats['layer_names']) * 0.3:
        print(f"{log_prefix}⚠️  WARNING: {grad_stats['num_zero_grads']}/{len(grad_stats['layer_names'])} layers have near-zero gradients!")

    if grad_stats['num_nan_grads'] > 0:
        print(f"{log_prefix}🚨 ERROR: {grad_stats['num_nan_grads']} layers have NaN gradients!")

    return grad_stats


# ============================================================================
# Trainer
# ============================================================================

def training_phase(model, dataloader, criterion_hc, criterion_pd, optimizer, device, gradient_accumulation_steps=1,
                   check_grads=False, max_grad_norm=1.0):
    """
    Training phase for one epoch with gradient accumulation.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion_hc: Loss function for HC vs PD task
        criterion_pd: Loss function for PD vs DD task
        optimizer: Optimizer
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating
        check_grads: Whether to check gradient statistics (for debugging)
        max_grad_norm: Maximum gradient norm for clipping (0 = no clipping)

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Zero gradients at the start
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        logits_hc_vs_pd, logits_pd_vs_dd = model(batch)

        # Calculate loss for HC vs PD (exclude -1 labels)
        valid_hc_mask = batch['hc_vs_pd'] != -1
        loss_hc = 0
        if valid_hc_mask.sum() > 0:
            loss_hc = criterion_hc(logits_hc_vs_pd[valid_hc_mask],
                               batch['hc_vs_pd'][valid_hc_mask])

        # Calculate loss for PD vs DD (exclude -1 labels)
        valid_pd_mask = batch['pd_vs_dd'] != -1
        loss_pd = 0
        if valid_pd_mask.sum() > 0:
            loss_pd = criterion_pd(logits_pd_vs_dd[valid_pd_mask],
                               batch['pd_vs_dd'][valid_pd_mask])

        # Combined loss
        loss = loss_hc + loss_pd

        # Scale loss by accumulation steps (to average gradients)
        loss = loss / gradient_accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        accumulated_loss += loss.item()
        total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
        num_batches += 1

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Check gradients on first few batches (debugging)
            if check_grads and batch_idx < 3:
                grad_stats = check_gradients(model, log_prefix=f"[Batch {batch_idx+1}] ")
                print(f"[Batch {batch_idx+1}] Total grad norm: {grad_stats['total_norm']:.4f}, "
                      f"Max: {grad_stats['max_grad']:.4f}, Min: {grad_stats['min_grad']:.4e}")

            # Gradient clipping (prevents exploding gradients)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({
                'loss': accumulated_loss * gradient_accumulation_steps,
                'eff_batch': (batch_idx + 1) // gradient_accumulation_steps
            })
            accumulated_loss = 0.0
        else:
            progress_bar.set_postfix({
                'loss': accumulated_loss * gradient_accumulation_steps,
                'accum': (batch_idx + 1) % gradient_accumulation_steps
            })

    # Update for any remaining accumulated gradients
    if num_batches % gradient_accumulation_steps != 0:
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss 

def validation_phase(model, dataloader, criterion_hc, criterion_pd, device, debug_patient_ids=False):
    """
    Validation phase.
    Returns validation loss and metrics for both tasks.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Collect predictions and labels for both tasks
    all_labels_hc = []
    all_preds_hc = []
    all_probs_hc = []

    all_labels_pd = []
    all_preds_pd = []
    all_probs_pd = []

    # Collect features for visualization
    all_features = []
    all_hc_pd_labels_viz = []
    all_pd_dd_labels_viz = []

    # Debug: Track patient IDs
    val_patient_ids = set()

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            logits_hc_vs_pd, logits_pd_vs_dd = model(batch)

            # Track patient IDs for debugging
            if debug_patient_ids and num_batches == 0:  # First batch only
                print("\n" + "="*60)
                print("VALIDATION BATCH STRUCTURE:")
                print("="*60)
                print(f"Batch keys: {list(batch.keys())}")
                print(f"'patient_ids' in batch: {'patient_ids' in batch}")
                if 'patient_ids' in batch:
                    print(f"Type of patient_ids: {type(batch['patient_ids'])}")
                    print(f"First 5 patient IDs: {batch['patient_ids'][:5]}")
                print("="*60)

            if 'patient_ids' in batch:
                val_patient_ids.update(batch['patient_ids'])

            # Extract features for visualization
            features_dict = model.get_features(batch)
            features = features_dict['fused_features'].cpu().numpy()
            all_features.append(features)
            all_hc_pd_labels_viz.append(batch['hc_vs_pd'].cpu().numpy())
            all_pd_dd_labels_viz.append(batch['pd_vs_dd'].cpu().numpy())

            # Calculate loss
            valid_hc_mask = batch['hc_vs_pd'] != -1
            loss_hc = 0
            if valid_hc_mask.sum() > 0:
                loss_hc = criterion_hc(logits_hc_vs_pd[valid_hc_mask],
                                   batch['hc_vs_pd'][valid_hc_mask])

            valid_pd_mask = batch['pd_vs_dd'] != -1
            loss_pd = 0
            if valid_pd_mask.sum() > 0:
                loss_pd = criterion_pd(logits_pd_vs_dd[valid_pd_mask],
                                   batch['pd_vs_dd'][valid_pd_mask])

            loss = loss_hc + loss_pd
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for HC vs PD
            if valid_hc_mask.sum() > 0:
                probs_hc = torch.softmax(logits_hc_vs_pd, dim=1)
                preds_hc = torch.argmax(logits_hc_vs_pd, dim=1)

                # Debug: Print first batch predictions
                if debug_patient_ids and num_batches == 1:
                    print("\n" + "="*60)
                    print("FIRST BATCH PREDICTIONS (HC vs PD):")
                    print("="*60)
                    print(f"Logits (first 5):\n{logits_hc_vs_pd[valid_hc_mask][:5].cpu().numpy()}")
                    print(f"\nTrue Labels (first 10): {batch['hc_vs_pd'][valid_hc_mask][:10].cpu().numpy()}")
                    print(f"Predictions (first 10): {preds_hc[valid_hc_mask][:10].cpu().numpy()}")
                    print(f"\nProbabilities for class 1 (first 5): {probs_hc[valid_hc_mask][:5, 1].cpu().numpy()}")
                    print("="*60)

                all_labels_hc.extend(batch['hc_vs_pd'][valid_hc_mask].cpu().numpy())
                all_preds_hc.extend(preds_hc[valid_hc_mask].cpu().numpy())
                all_probs_hc.extend(probs_hc[valid_hc_mask, 1].cpu().numpy())  # Probability of positive class

            # Collect predictions for PD vs DD
            if valid_pd_mask.sum() > 0:
                probs_pd = torch.softmax(logits_pd_vs_dd, dim=1)
                preds_pd = torch.argmax(logits_pd_vs_dd, dim=1)

                all_labels_pd.extend(batch['pd_vs_dd'][valid_pd_mask].cpu().numpy())
                all_preds_pd.extend(preds_pd[valid_pd_mask].cpu().numpy())
                all_probs_pd.extend(probs_pd[valid_pd_mask, 1].cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Calculate metrics for HC vs PD
    metrics_hc = {}
    if len(all_labels_hc) > 0:
        metrics_hc = calculate_metrics(all_labels_hc, all_preds_hc, task_name="HC vs PD", verbose=True)
        metrics_hc['labels'] = all_labels_hc
        metrics_hc['predictions'] = all_preds_hc
        metrics_hc['probabilities'] = all_probs_hc

    # Calculate metrics for PD vs DD
    metrics_pd = {}
    if len(all_labels_pd) > 0:
        metrics_pd = calculate_metrics(all_labels_pd, all_preds_pd, task_name="PD vs DD", verbose=True)
        metrics_pd['labels'] = all_labels_pd
        metrics_pd['predictions'] = all_preds_pd
        metrics_pd['probabilities'] = all_probs_pd

    # Combine features and labels for visualization
    if len(all_features) > 0:
        all_features = np.concatenate(all_features, axis=0)
        all_hc_pd_labels_viz = np.concatenate(all_hc_pd_labels_viz, axis=0)
        all_pd_dd_labels_viz = np.concatenate(all_pd_dd_labels_viz, axis=0)
    else:
        all_features = None
        all_hc_pd_labels_viz = None
        all_pd_dd_labels_viz = None

    # Debug patient IDs
    if debug_patient_ids:
        print("\n" + "="*60)
        print("VALIDATION SET PATIENT IDs:")
        print("="*60)
        print(f"Total unique patients in validation: {len(val_patient_ids)}")
        if len(val_patient_ids) > 0:
            print(f"Sample patient IDs: {sorted(list(val_patient_ids))[:10]}")
        else:
            print("[ERROR] No patient IDs collected from validation set!")
        print("="*60)

    return avg_loss, metrics_hc, metrics_pd, all_features, all_hc_pd_labels_viz, all_pd_dd_labels_viz, val_patient_ids

def train_model(config):
    """
    Train model using k-fold cross-validation.
    """
    print("\n" + "="*80)
    print("STARTING TRAINING WITH K-FOLD CROSS-VALIDATION")
    print("="*80)

    # Load full dataset
    print("\nLoading dataset...")
    full_dataset = ParkinsonsDatasetLoader(
        data_root=config['data_root'],
        window_size=config['seq_len'],
        max_windows_per_task=10,
        min_windows_per_task=2,
        apply_downsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )

    # Get k-fold splits
    print(f"\nCreating {config['num_folds']}-fold cross-validation splits...")
    fold_datasets = full_dataset.get_k_fold_split(k=config['num_folds'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Store results for all folds
    all_fold_results = []

    # Train on each fold
    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/{config['num_folds']}")
        print("="*80)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn
        )

        # Initialize model
        model = MyModel(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_window_layers=config['num_window_layers'],
            num_task_layers=config['num_task_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            seq_len=config['seq_len']
        ).to(device)

        # Compute class weights for handling imbalance (better than differential sampling)
        # Count labels in training data
        train_hc_count = sum(1 for pt in train_dataset.patient_task_data if pt['hc_vs_pd'] == 0)
        train_pd_count = sum(1 for pt in train_dataset.patient_task_data if pt['hc_vs_pd'] == 1)
        train_dd_count = sum(1 for pt in train_dataset.patient_task_data if pt['pd_vs_dd'] == 1)

        # Calculate inverse frequency weights
        total_hc_pd = train_hc_count + train_pd_count
        total_pd_dd = train_pd_count + train_dd_count

        weight_hc_pd = torch.FloatTensor([
            total_hc_pd / (2 * train_hc_count),
            total_hc_pd / (2 * train_pd_count)
        ]).to(device)

        weight_pd_dd = torch.FloatTensor([
            total_pd_dd / (2 * train_pd_count),
            total_pd_dd / (2 * train_dd_count)
        ]).to(device)

        print(f"\n[Class Weighting] HC vs PD weights: HC={weight_hc_pd[0]:.3f}, PD={weight_hc_pd[1]:.3f}")
        print(f"[Class Weighting] PD vs DD weights: PD={weight_pd_dd[0]:.3f}, DD={weight_pd_dd[1]:.3f}")

        # Use weighted loss instead of differential sampling
        criterion_hc = nn.CrossEntropyLoss(weight=weight_hc_pd)
        criterion_pd = nn.CrossEntropyLoss(weight=weight_pd_dd)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # ============ COLLECT TRAIN PATIENT IDS FOR DEBUGGING ============
        print("\n" + "="*60)
        print("COLLECTING TRAIN PATIENT IDs FOR DEBUGGING...")
        print("="*60)
        train_patient_ids = set()
        batch_count = 0
        for batch in train_loader:
            if batch_count == 0:  # First batch debug
                print(f"[DEBUG] First TRAIN batch keys: {list(batch.keys())}")
                print(f"[DEBUG] 'patient_ids' in batch: {'patient_ids' in batch}")
                if 'patient_ids' in batch:
                    print(f"[DEBUG] Type of patient_ids: {type(batch['patient_ids'])}")
                    print(f"[DEBUG] First 5 patient IDs: {batch['patient_ids'][:5]}")
            if 'patient_ids' in batch:
                train_patient_ids.update(batch['patient_ids'])
            batch_count += 1

        print(f"\n[RESULT] Training set has {len(train_patient_ids)} unique patients")
        if len(train_patient_ids) > 0:
            print(f"[RESULT] Sample train patient IDs: {sorted(list(train_patient_ids))[:10]}")
        else:
            print(f"[ERROR] No patient IDs collected from training set!")
        print("="*60 + "\n")
        # ==================================================================

        # Track best model for this fold
        best_val_acc = 0.0
        best_epoch = 0
        best_metrics_hc = {}
        best_metrics_pd = {}
        best_features = None
        best_hc_labels = None
        best_pd_labels = None

        # Store metrics for each epoch
        epoch_metrics_hc = []
        epoch_metrics_pd = []

        # Training loop
        num_epochs = config['num_epochs']
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

            # Training phase (check gradients on first epoch for debugging)
            train_loss = training_phase(
                model, train_loader, criterion_hc, criterion_pd, optimizer, device,
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                check_grads=(epoch == 0 and fold_idx == 0),  # Only first epoch of first fold
                max_grad_norm=config.get('max_grad_norm', 1.0)
            )

            # Validation phase (debug patient IDs on first epoch)
            val_loss, val_metrics_hc, val_metrics_pd, features, hc_labels, pd_labels, val_patient_ids = validation_phase(
                model, val_loader, criterion_hc, criterion_pd, device, debug_patient_ids=(epoch == 0)
            )

            # ============ CHECK FOR DATA LEAKAGE ON FIRST EPOCH ============
            if epoch == 0:
                print("\n" + "="*70)
                print("DATA LEAKAGE CHECK - COMPARING TRAIN AND VALIDATION PATIENTS")
                print("="*70)

                overlap = train_patient_ids.intersection(val_patient_ids)

                print(f"Train patients: {len(train_patient_ids)}")
                print(f"Validation patients: {len(val_patient_ids)}")
                print(f"Overlapping patients: {len(overlap)}")

                if overlap:
                    print(f"\n🚨🚨🚨 CRITICAL: DATA LEAKAGE DETECTED! 🚨🚨🚨")
                    print(f"   {len(overlap)} patients appear in BOTH train and validation!")
                    print(f"   Overlapping IDs: {sorted(list(overlap))[:20]}")
                    print(f"   This EXPLAINS the 100% accuracy - MODEL IS CHEATING!")
                    print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                else:
                    print(f"\n✓✓✓ GOOD NEWS: No train/val patient overlap detected!")

                    # Check for subtle leakage via num_windows
                    print(f"\n🔍 Checking for num_windows leakage...")
                    # Collect num_windows statistics from validation loader
                    num_windows_by_label_hc = {0: [], 1: []}
                    num_windows_by_label_pd = {0: [], 1: []}

                    for batch in val_loader:
                        for i, nw in enumerate(batch['num_windows']):
                            if batch['hc_vs_pd'][i].item() != -1:
                                num_windows_by_label_hc[batch['hc_vs_pd'][i].item()].append(nw)
                            if batch['pd_vs_dd'][i].item() != -1:
                                num_windows_by_label_pd[batch['pd_vs_dd'][i].item()].append(nw)

                    # Analyze HC vs PD
                    if len(num_windows_by_label_hc[0]) > 0 and len(num_windows_by_label_hc[1]) > 0:
                        avg_hc = np.mean(num_windows_by_label_hc[0])
                        avg_pd = np.mean(num_windows_by_label_hc[1])
                        print(f"   HC vs PD - Avg windows: HC={avg_hc:.2f}, PD={avg_pd:.2f}")
                        if abs(avg_hc - avg_pd) > 2:
                            print(f"   ⚠️  SUSPICIOUS: Large difference in num_windows!")
                            print(f"   Model might be using window count to classify!")

                    # Analyze PD vs DD
                    if len(num_windows_by_label_pd[0]) > 0 and len(num_windows_by_label_pd[1]) > 0:
                        avg_pd2 = np.mean(num_windows_by_label_pd[0])
                        avg_dd = np.mean(num_windows_by_label_pd[1])
                        print(f"   PD vs DD - Avg windows: PD={avg_pd2:.2f}, DD={avg_dd:.2f}")
                        if abs(avg_pd2 - avg_dd) > 2:
                            print(f"   ⚠️  SUSPICIOUS: Large difference in num_windows!")
                            print(f"   Model might be using window count to classify!")

                    # Additional diagnostic
                    if len(train_patient_ids) == 0 or len(val_patient_ids) == 0:
                        print(f"\n⚠️  WARNING: Patient IDs not being collected properly!")
                        print(f"   Train IDs collected: {len(train_patient_ids) > 0}")
                        print(f"   Val IDs collected: {len(val_patient_ids) > 0}")
                        print(f"   Can't verify absence of data leakage!")
                        print(f"   100% accuracy might be due to undetected leakage.")

                print("="*70 + "\n")
            # ===============================================================

            # Calculate average accuracy across both tasks
            acc_hc = val_metrics_hc.get('accuracy', 0) if val_metrics_hc else 0
            acc_pd = val_metrics_pd.get('accuracy', 0) if val_metrics_pd else 0
            avg_acc = (acc_hc + acc_pd) / 2

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg Acc: {avg_acc:.4f}")

            # Store metrics for this epoch
            if val_metrics_hc:
                epoch_metrics_hc.append({
                    'epoch': epoch + 1,
                    'metrics': {
                        'accuracy': acc_hc,
                        'precision': val_metrics_hc.get('precision_avg', 0),
                        'recall': val_metrics_hc.get('recall_avg', 0),
                        'f1': val_metrics_hc.get('f1_avg', 0)
                    }
                })

            if val_metrics_pd:
                epoch_metrics_pd.append({
                    'epoch': epoch + 1,
                    'metrics': {
                        'accuracy': acc_pd,
                        'precision': val_metrics_pd.get('precision_avg', 0),
                        'recall': val_metrics_pd.get('recall_avg', 0),
                        'f1': val_metrics_pd.get('f1_avg', 0)
                    }
                })

            # Save best model
            if avg_acc > best_val_acc:
                best_val_acc = avg_acc
                best_epoch = epoch + 1
                best_metrics_hc = val_metrics_hc
                best_metrics_pd = val_metrics_pd
                best_features = features
                best_hc_labels = hc_labels
                best_pd_labels = pd_labels

                # Save model checkpoint
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx,
                    'accuracy': avg_acc
                }, f"checkpoints/best_model_fold_{fold_idx+1}.pt")
                print(f"✓ New best model saved (Acc: {best_val_acc:.4f})")

        # Save metrics for this fold
        print(f"\n--- Fold {fold_idx+1} Complete ---")
        print(f"Best Epoch: {best_epoch} | Best Accuracy: {best_val_acc:.4f}")

        if config['save_metrics']:
            fold_suffix = f"_fold_{fold_idx+1}"
            save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                           epoch_metrics_hc, epoch_metrics_pd)

        # Create plots for best epoch
        if config['create_plots']:
            plot_dir = f"plots/fold_{fold_idx+1}"
            os.makedirs(plot_dir, exist_ok=True)

            # Plot ROC curves
            if best_metrics_hc and 'labels' in best_metrics_hc:
                plot_roc_curves(
                    best_metrics_hc['labels'],
                    best_metrics_hc['predictions'],
                    best_metrics_hc['probabilities'],
                    os.path.join(plot_dir, "roc_hc_vs_pd.png")
                )
                print(f"✓ ROC curve saved: {plot_dir}/roc_hc_vs_pd.png")

            if best_metrics_pd and 'labels' in best_metrics_pd:
                plot_roc_curves(
                    best_metrics_pd['labels'],
                    best_metrics_pd['predictions'],
                    best_metrics_pd['probabilities'],
                    os.path.join(plot_dir, "roc_pd_vs_dd.png")
                )
                print(f"✓ ROC curve saved: {plot_dir}/roc_pd_vs_dd.png")

            # Plot t-SNE
            if best_features is not None:
                plot_tsne(best_features, best_hc_labels, best_pd_labels, output_dir=plot_dir)

        # Store fold results
        all_fold_results.append({
            'fold': fold_idx + 1,
            'best_epoch': best_epoch,
            'best_accuracy': best_val_acc,
            'metrics_hc': best_metrics_hc,
            'metrics_pd': best_metrics_pd
        })

    # Print summary of all folds
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    for result in all_fold_results:
        print(f"Fold {result['fold']}: Best Epoch={result['best_epoch']}, Accuracy={result['best_accuracy']:.4f}")

    avg_accuracy = np.mean([r['best_accuracy'] for r in all_fold_results])
    std_accuracy = np.std([r['best_accuracy'] for r in all_fold_results])
    print(f"\nAverage Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    return all_fold_results

def main():
    config = get_config()

    results = train_model(config)

    return results


if __name__ == "__main__":
    results = main()