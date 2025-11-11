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
        'learning_rate': 0.001,  # Increased from 0.0005 to escape local minimum
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,
        'max_grad_norm': 1.0,  # Gradient clipping threshold (0 = no clipping)
        'use_auxiliary_loss': False,  # Enable if vanishing gradients detected (adds window-level supervision)
        'label_smoothing': 0.1,  # Mild label smoothing (oversampling handles imbalance)

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

        # Store patient data with ALL tasks grouped together
        # Each entry: {patient_id, tasks: [list of task dicts], hc_vs_pd, pd_vs_dd}
        # where each task dict has: {task_name, left_windows, right_windows, num_windows}
        self.patient_data = []
        
        if data_root is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"

            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            
            self.patient_ids_list = list(range(1, 470))
            print(f"Loading dataset: {len(self.patient_ids_list)} patients")
            
            self._load_data()
        elif patient_task_data is not None:
            # For split datasets - backward compatibility
            self.patient_data = patient_task_data

        print(f"Total patients: {len(self.patient_data)}")
    
    
    def _load_data(self):
        """Load data organized by patient with ALL tasks grouped together"""

        for patient_id in tqdm(self.patient_ids_list, desc="Loading patient data"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))

            if not patient_path.exists():
                continue

            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)

                condition = metadata.get('condition', '')

                # Determine labels and differential overlapping
                if condition == 'Healthy':
                    overlap = 0.70
                    hc_vs_pd_label = 0
                    pd_vs_dd_label = -1
                elif 'Parkinson' in condition:
                    overlap = 0.0
                    hc_vs_pd_label = 1
                    pd_vs_dd_label = 0
                else:
                    overlap = 0.65
                    hc_vs_pd_label = -1
                    pd_vs_dd_label = 1

                # Collect all tasks for this patient
                patient_tasks = []

                # Process each task and collect them
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

                                # Add this task to patient's task list
                                patient_tasks.append({
                                    'task_name': task,
                                    'left_windows': left_windows[:num_windows],  # shape: (num_windows, 256, 6)
                                    'right_windows': right_windows[:num_windows], # shape: (num_windows, 256, 6)
                                    'num_windows': num_windows
                                })
                    
                    except Exception as e:
                        print(f"Error loading patient {patient_id}, task {task}: {e}")
                        continue

                # After processing all tasks, add patient if they have at least one valid task
                if len(patient_tasks) > 0:
                    self.patient_data.append({
                        'patient_id': patient_id,
                        'tasks': patient_tasks,
                        'hc_vs_pd': hc_vs_pd_label,
                        'pd_vs_dd': pd_vs_dd_label,
                        'num_tasks': len(patient_tasks)
                    })

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

            # Split patient data
            train_data = [pt for pt in self.patient_data if pt['patient_id'] in train_patients]
            test_data = [pt for pt in self.patient_data if pt['patient_id'] in test_patients]
            
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
            print(f"  Train: {len(train_data)} patients (HC={train_hc}, PD={train_pd}, DD={train_dd})")
            print(f"  Test:  {len(test_data)} patients (HC={test_hc}, PD={test_pd}, DD={test_dd})")
            
            fold_datasets.append((train_dataset, test_dataset))
        
        return fold_datasets
    
    
    def __len__(self):
        return len(self.patient_data)


    def __getitem__(self, idx):
        """Returns ONE patient with ALL their tasks."""
        patient = self.patient_data[idx]

        # Convert all task windows to tensors
        tasks_data = []
        for task in patient['tasks']:
            tasks_data.append({
                'task_name': task['task_name'],
                'left_windows': torch.FloatTensor(task['left_windows']),   # (num_windows, 256, 6)
                'right_windows': torch.FloatTensor(task['right_windows']), # (num_windows, 256, 6)
                'num_windows': task['num_windows']
            })

        return {
            'patient_id': patient['patient_id'],
            'tasks': tasks_data,
            'num_tasks': patient['num_tasks'],
            'hc_vs_pd': patient['hc_vs_pd'],
            'pd_vs_dd': patient['pd_vs_dd']
        }


# ============================================================================
# Custom collate function for variable-length sequences
# ============================================================================
def collate_fn(batch):
    """
    Collate function to handle variable number of tasks per patient and windows per task.
    Pads to max_tasks and max_windows in the batch.
    Structure: batch_size x max_tasks x max_windows x 256 x 6
    """
    batch_size = len(batch)

    # Find max tasks and max windows across all patients in batch
    max_tasks = max([patient['num_tasks'] for patient in batch])
    max_windows = max([
        max([task['num_windows'] for task in patient['tasks']])
        for patient in batch
    ])

    # Initialize padded tensors
    # Shape: (batch_size, max_tasks, max_windows, 256, 6)
    left_windows_padded = torch.zeros(batch_size, max_tasks, max_windows, 256, 6)
    right_windows_padded = torch.zeros(batch_size, max_tasks, max_windows, 256, 6)

    # Masks: True for valid positions
    task_masks = torch.zeros(batch_size, max_tasks, dtype=torch.bool)     # Valid tasks
    window_masks = torch.zeros(batch_size, max_tasks, max_windows, dtype=torch.bool)  # Valid windows

    hc_vs_pd_labels = []
    pd_vs_dd_labels = []
    patient_ids = []

    for i, patient in enumerate(batch):
        num_tasks = patient['num_tasks']
        task_masks[i, :num_tasks] = True

        for j, task in enumerate(patient['tasks']):
            num_windows = task['num_windows']

            # Copy actual window data
            left_windows_padded[i, j, :num_windows] = task['left_windows']
            right_windows_padded[i, j, :num_windows] = task['right_windows']
            window_masks[i, j, :num_windows] = True

        hc_vs_pd_labels.append(patient['hc_vs_pd'])
        pd_vs_dd_labels.append(patient['pd_vs_dd'])
        patient_ids.append(patient['patient_id'])

    return {
        'left_windows': left_windows_padded,      # (batch, max_tasks, max_windows, 256, 6)
        'right_windows': right_windows_padded,    # (batch, max_tasks, max_windows, 256, 6)
        'task_masks': task_masks,                  # (batch, max_tasks)
        'window_masks': window_masks,              # (batch, max_tasks, max_windows)
        'hc_vs_pd': torch.LongTensor(hc_vs_pd_labels),
        'pd_vs_dd': torch.LongTensor(pd_vs_dd_labels),
        'patient_ids': patient_ids
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
        use_auxiliary_loss: bool = False,  # Add auxiliary loss at window level if vanishing gradients detected
    ):
        super().__init__()

        self.model_dim = model_dim
        self.seq_len = seq_len
        self.use_auxiliary_loss = use_auxiliary_loss

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
        # After fusion, we have model_dim features per window (not model_dim*2)
        self.window_positional_encoding = PositionalEncoding(model_dim, max_len=100)  # max 100 windows

        self.task_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(model_dim),
                'feed_forward': FeedForward(model_dim, d_ff, dropout)
            })
            for _ in range(num_task_layers)
        ])

        # Task-level pooling (attention mechanism)
        # Note: No bias in attention scoring - bias doesn't affect softmax output
        self.task_attention_pooling = nn.Linear(model_dim, 1, bias=False)

        # ========== Classification Heads ==========
        # After fusion: using model_dim features (not model_dim*2)
        fusion_dim = model_dim

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

        # ========== Auxiliary Classification Heads (for vanishing gradient mitigation) ==========
        # These operate directly on task features, providing shorter gradient path
        if use_auxiliary_loss:
            self.aux_head_hc_vs_pd = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 2)
            )

            self.aux_head_pd_vs_dd = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 2)
            )

        self.dropout = nn.Dropout(dropout)

    def initialize_classifier_bias(self, train_dataset):
        """
        Initialize classification head biases based on class frequencies.
        This gives minority classes a better starting point.
        """
        # Count class frequencies in training data
        hc_count = sum(1 for pt in train_dataset.patient_data if pt['hc_vs_pd'] == 0)
        pd_count = sum(1 for pt in train_dataset.patient_data if pt['hc_vs_pd'] == 1)
        dd_count = sum(1 for pt in train_dataset.patient_data if pt['pd_vs_dd'] == 1)

        # HC vs PD bias initialization
        total_hc_pd = hc_count + pd_count
        if total_hc_pd > 0:
            freq_hc = hc_count / total_hc_pd
            freq_pd = pd_count / total_hc_pd
            # Initialize bias to log frequencies (makes initial predictions match class distribution)
            with torch.no_grad():
                self.head_hc_vs_pd[-1].bias[0] = np.log(freq_hc + 1e-6)
                self.head_hc_vs_pd[-1].bias[1] = np.log(freq_pd + 1e-6)
            print(f"[Bias Init] HC vs PD: HC={freq_hc:.3f} (bias={np.log(freq_hc + 1e-6):.3f}), "
                  f"PD={freq_pd:.3f} (bias={np.log(freq_pd + 1e-6):.3f})")

        # PD vs DD bias initialization
        total_pd_dd = pd_count + dd_count
        if total_pd_dd > 0:
            freq_pd2 = pd_count / total_pd_dd
            freq_dd = dd_count / total_pd_dd
            with torch.no_grad():
                self.head_pd_vs_dd[-1].bias[0] = np.log(freq_pd2 + 1e-6)
                self.head_pd_vs_dd[-1].bias[1] = np.log(freq_dd + 1e-6)
            print(f"[Bias Init] PD vs DD: PD={freq_pd2:.3f} (bias={np.log(freq_pd2 + 1e-6):.3f}), "
                  f"DD={freq_dd:.3f} (bias={np.log(freq_dd + 1e-6):.3f})")


    def forward(self, batch):
        """
        Forward pass for hierarchical model.
        Input: batch_size x max_tasks x max_windows x 256 x 6
        Output: batch_size x num_classes
        """
        device = batch['left_windows'].device
        batch_size = batch['left_windows'].shape[0]
        max_tasks = batch['left_windows'].shape[1]
        max_windows = batch['left_windows'].shape[2]

        # ========== LEVEL 1: Window-Level Cross-Attention (within each task) ==========
        # Reshape: (batch, max_tasks, max_windows, 256, 6) -> (batch * max_tasks * max_windows, 256, 6)
        left_windows_flat = batch['left_windows'].view(-1, self.seq_len, 6)
        right_windows_flat = batch['right_windows'].view(-1, self.seq_len, 6)
        
        # Project to model dimension
        left_encoded = self.left_projection(left_windows_flat)   # (batch*max_tasks*max_windows, 256, model_dim)
        right_encoded = self.right_projection(right_windows_flat)  # (batch*max_tasks*max_windows, 256, model_dim) 
        
        # Add positional encoding
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        # Apply cross-attention layers between left and right wrist
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        # Fuse left and right after cross-attention into single representation
        # Cross-attention has already exchanged information, now unify them
        fused_encoded = (left_encoded + right_encoded) / 2  # (batch*max_tasks*max_windows, 256, model_dim)

        # Global pooling for each window (now pooling fused representation)
        window_features = self.global_pool(fused_encoded.transpose(1, 2)).squeeze(-1)  # (batch*max_tasks*max_windows, model_dim)

        # Reshape to (batch, max_tasks, max_windows, model_dim)
        window_features = window_features.view(batch_size, max_tasks, max_windows, -1)

        # ========== LEVEL 1.5: Aggregate windows within each task ==========
        # For each task, aggregate its windows using attention
        task_features_list = []

        for task_idx in range(max_tasks):
            task_windows = window_features[:, task_idx, :, :]  # (batch, max_windows, model_dim)
            task_window_mask = batch['window_masks'][:, task_idx, :]  # (batch, max_windows)

            # Shuffle windows within this task to prevent learning from serial order
            if self.training:
                shuffled_task_windows = []
                shuffled_task_masks = []
                for i in range(batch_size):
                    perm = torch.randperm(max_windows, device=window_features.device)
                    shuffled_task_windows.append(task_windows[i][perm].unsqueeze(0))
                    shuffled_task_masks.append(task_window_mask[i][perm].unsqueeze(0))
                task_windows = torch.cat(shuffled_task_windows, dim=0)
                task_window_mask = torch.cat(shuffled_task_masks, dim=0)

            # Apply attention pooling to aggregate windows for this task
            task_windows_with_pe = self.window_positional_encoding(task_windows)

            # Attention scores for window aggregation
            attention_scores = self.task_attention_pooling(task_windows_with_pe)  # (batch, max_windows, 1)
            attention_scores = attention_scores.masked_fill(~task_window_mask.unsqueeze(-1), float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)

            # Aggregate windows → single task representation
            task_repr = (task_windows * attention_weights).sum(dim=1)  # (batch, model_dim)
            task_features_list.append(task_repr)

        # Stack all task representations: (batch, max_tasks, model_dim)
        task_features = torch.stack(task_features_list, dim=1)

        # ========== LEVEL 2: Aggregate across tasks per patient ==========

        # Apply self-attention across tasks
        task_mask = ~batch['task_masks']  # (batch, max_tasks) - True for padding

        # Apply task-level self-attention layers
        for task_layer in self.task_layers:
            attn_output, _ = task_layer['self_attention'](
                query=task_features,
                key=task_features,
                value=task_features,
                key_padding_mask=task_mask
            )
            task_features = task_layer['norm'](task_features + attn_output)
            task_features = task_layer['feed_forward'](task_features)

        # ========== Auxiliary Loss (if enabled) - Shorter Gradient Path ==========
        aux_logits_hc = None
        aux_logits_pd = None
        if self.use_auxiliary_loss:
            # Pool task features for auxiliary classification (simple average over valid tasks)
            valid_task_mask = batch['task_masks'].float().unsqueeze(-1)  # (batch, max_tasks, 1)
            num_valid_tasks = valid_task_mask.sum(dim=1, keepdim=True).clamp(min=1)
            task_features_for_aux = (task_features * valid_task_mask).sum(dim=1) / num_valid_tasks.squeeze(-1)  # (batch, model_dim)
            aux_logits_hc = self.aux_head_hc_vs_pd(task_features_for_aux)
            aux_logits_pd = self.aux_head_pd_vs_dd(task_features_for_aux)

        # Final attention pooling across tasks → patient representation
        attention_scores = self.task_attention_pooling(task_features)  # (batch, max_tasks, 1)
        attention_scores = attention_scores.masked_fill(task_mask.unsqueeze(-1), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, max_tasks, 1)

        # Weighted sum of task features → patient representation
        patient_representation = (task_features * attention_weights).sum(dim=1)  # (batch, model_dim)

        # ========== Classification ==========
        # Using patient representation aggregated from all tasks
        logits_hc_vs_pd = self.head_hc_vs_pd(patient_representation)
        logits_pd_vs_dd = self.head_pd_vs_dd(patient_representation)

        # Return main logits and auxiliary logits (if enabled)
        if self.use_auxiliary_loss:
            return logits_hc_vs_pd, logits_pd_vs_dd, aux_logits_hc, aux_logits_pd
        else:
            return logits_hc_vs_pd, logits_pd_vs_dd
    
    
    def get_features(self, batch):
        """
        Extract patient-level features for visualization (e.g., t-SNE).
        Uses forward pass without gradients.
        """
        with torch.no_grad():
            # Run forward pass
            if self.use_auxiliary_loss:
                logits_hc, logits_pd, aux_hc, aux_pd = self.forward(batch)
            else:
                logits_hc, logits_pd = self.forward(batch)

            # Return logits as features for visualization
            # These represent the patient-level representation after aggregating all tasks
            patient_features = torch.cat([logits_hc, logits_pd], dim=1)  # (batch, 4) - concatenated logits
            return {
                'patient_features': patient_features,
                'fused_features': patient_features,  # Backward compatibility
                'logits_hc_vs_pd': logits_hc,  # (batch, 2)
                'logits_pd_vs_dd': logits_pd   # (batch, 2)
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

def plot_loss(train_losses, val_losses, fold_idx=None, output_dir="plots"):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        fold_idx: Optional fold number for filename
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot loss difference (overfitting indicator)
    plt.subplot(1, 2, 2)
    loss_diff = [val - train for train, val in zip(train_losses, val_losses)]
    plt.plot(epochs, loss_diff, 'g-', label='Val Loss - Train Loss', linewidth=2, marker='d', markersize=4)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Difference', fontsize=12)
    plt.title('Overfitting Indicator (Val - Train)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add annotation for best epoch (lowest validation loss)
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.subplot(1, 2, 1)
    plt.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5, marker='*')
    plt.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    output_path = os.path.join(output_dir, f"loss_curves{fold_suffix}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Loss curves saved: {output_path}")
    print(f"  Best epoch: {best_epoch} with validation loss: {best_val_loss:.4f}")

    # Also create a detailed loss log
    log_path = os.path.join(output_dir, f"loss_log{fold_suffix}.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Loss Difference'])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_loss - train_loss:.6f}"])

    print(f"✓ Loss log saved: {log_path}")

    return best_epoch, best_val_loss

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


def analyze_hierarchical_gradient_flow(model):
    """
    Detailed gradient flow analysis for hierarchical architecture.
    Checks gradients at each level of the hierarchy.
    """
    print("\n" + "="*80)
    print("HIERARCHICAL GRADIENT FLOW ANALYSIS")
    print("="*80)

    # Organize layers by hierarchy level
    window_level_grads = []
    task_level_grads = []
    pooling_grads = []
    classification_grads = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.data.norm(2).item()

        # Categorize by hierarchy level
        if 'window_layers' in name or 'left_projection' in name or 'right_projection' in name or 'positional_encoding' in name:
            window_level_grads.append((name, grad_norm))
        elif 'task_layers' in name or 'window_positional_encoding' in name:
            task_level_grads.append((name, grad_norm))
        elif 'task_attention_pooling' in name or 'global_pool' in name:
            pooling_grads.append((name, grad_norm))
        elif 'head_' in name:
            classification_grads.append((name, grad_norm))

    # Analyze each level
    print("\n🔵 LEVEL 1: Window-Level Processing (deepest in hierarchy)")
    print("-" * 80)
    if window_level_grads:
        avg_grad = sum(g for _, g in window_level_grads) / len(window_level_grads)
        max_grad = max(g for _, g in window_level_grads)
        min_grad = min(g for _, g in window_level_grads)
        zero_count = sum(1 for _, g in window_level_grads if g < 1e-7)

        print(f"Avg gradient: {avg_grad:.6f} | Max: {max_grad:.6f} | Min: {min_grad:.2e}")
        print(f"Layers with near-zero gradients: {zero_count}/{len(window_level_grads)}")

        if avg_grad < 1e-4:
            print("🚨 CRITICAL: Window-level gradients are vanishing!")
        elif avg_grad < 1e-3:
            print("⚠️  WARNING: Window-level gradients are very small")
        else:
            print("✅ Window-level gradients look healthy")

        # Show worst layers
        worst_layers = sorted(window_level_grads, key=lambda x: x[1])[:3]
        print("\nLayers with smallest gradients:")
        for name, grad in worst_layers:
            print(f"  {name}: {grad:.2e}")

    print("\n🟢 LEVEL 2: Task-Level Processing")
    print("-" * 80)
    if task_level_grads:
        avg_grad = sum(g for _, g in task_level_grads) / len(task_level_grads)
        max_grad = max(g for _, g in task_level_grads)
        min_grad = min(g for _, g in task_level_grads)

        print(f"Avg gradient: {avg_grad:.6f} | Max: {max_grad:.6f} | Min: {min_grad:.2e}")

        if avg_grad < 1e-4:
            print("🚨 CRITICAL: Task-level gradients are vanishing!")
        elif avg_grad < 1e-3:
            print("⚠️  WARNING: Task-level gradients are very small")
        else:
            print("✅ Task-level gradients look healthy")

    print("\n🟡 POOLING LAYERS (potential bottleneck)")
    print("-" * 80)
    if pooling_grads:
        for name, grad in pooling_grads:
            print(f"{name}: {grad:.6f}")
            if grad < 1e-5:
                print("  🚨 CRITICAL: Pooling layer blocking gradients!")

    print("\n🟠 CLASSIFICATION HEADS (closest to loss)")
    print("-" * 80)
    if classification_grads:
        avg_grad = sum(g for _, g in classification_grads) / len(classification_grads)
        print(f"Avg gradient: {avg_grad:.6f}")

        for name, grad in classification_grads:
            print(f"  {name}: {grad:.6f}")

    # Gradient flow ratio analysis
    print("\n📊 GRADIENT FLOW RATIO ANALYSIS")
    print("-" * 80)
    if window_level_grads and classification_grads:
        window_avg = sum(g for _, g in window_level_grads) / len(window_level_grads)
        class_avg = sum(g for _, g in classification_grads) / len(classification_grads)
        ratio = window_avg / (class_avg + 1e-10)

        print(f"Window-to-Classification gradient ratio: {ratio:.6f}")
        print(f"Window avg: {window_avg:.6f} | Classification avg: {class_avg:.6f}")

        if ratio < 0.001:
            print("🚨 SEVERE VANISHING: Window gradients are 1000x smaller than classification!")
            print("   RECOMMENDATION: Enable auxiliary loss in config:")
            print("   Set 'use_auxiliary_loss': True")
            print("   This adds window-level supervision with shorter gradient path")
        elif ratio < 0.01:
            print("⚠️  MODERATE VANISHING: Window gradients are 100x smaller than classification")
            print("   RECOMMENDATION: Consider one of:")
            print("   1. Enable 'use_auxiliary_loss': True (adds window-level supervision)")
            print("   2. Reduce 'num_window_layers' from 4 to 2-3")
            print("   3. Increase learning rate slightly")
        elif ratio < 0.1:
            print("⚠️  MILD VANISHING: Window gradients are 10x smaller than classification")
            print("   May be acceptable, but monitor training closely")
            print("   If training stalls, enable 'use_auxiliary_loss': True")
        else:
            print("✅ Good gradient flow through hierarchy - no action needed!")

    # Check for zero-gradient bias in pooling (indicates attention collapse)
    pooling_bias_grad = None
    for name, grad in pooling_grads:
        if 'bias' in name and grad < 1e-10:
            print(f"\n🚨 ATTENTION COLLAPSE DETECTED!")
            print(f"   {name} has zero gradient - attention weights may be uniform")
            print(f"   RECOMMENDATIONS:")
            print(f"   1. Increase learning rate: try 0.001 or 0.002")
            print(f"   2. Use stronger class weights (already applied)")
            print(f"   3. Consider using focal loss for severe imbalance")

    print("="*80 + "\n")


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
        model_outputs = model(batch)
        if len(model_outputs) == 4:  # Auxiliary loss enabled
            logits_hc_vs_pd, logits_pd_vs_dd, aux_logits_hc, aux_logits_pd = model_outputs
        else:  # Standard forward pass
            logits_hc_vs_pd, logits_pd_vs_dd = model_outputs
            aux_logits_hc, aux_logits_pd = None, None

        # Calculate loss for HC vs PD (exclude -1 labels)
        valid_hc_mask = batch['hc_vs_pd'] != -1
        loss_hc = 0
        if valid_hc_mask.sum() > 0:
            loss_hc = criterion_hc(logits_hc_vs_pd[valid_hc_mask],
                               batch['hc_vs_pd'][valid_hc_mask])

            # Add auxiliary loss if enabled (weight it lower than main loss)
            if aux_logits_hc is not None:
                aux_loss_hc = criterion_hc(aux_logits_hc[valid_hc_mask],
                                          batch['hc_vs_pd'][valid_hc_mask])
                loss_hc = loss_hc + 0.4 * aux_loss_hc  # 40% weight for auxiliary

        # Calculate loss for PD vs DD (exclude -1 labels)
        valid_pd_mask = batch['pd_vs_dd'] != -1
        loss_pd = 0
        if valid_pd_mask.sum() > 0:
            loss_pd = criterion_pd(logits_pd_vs_dd[valid_pd_mask],
                               batch['pd_vs_dd'][valid_pd_mask])

            # Add auxiliary loss if enabled
            if aux_logits_pd is not None:
                aux_loss_pd = criterion_pd(aux_logits_pd[valid_pd_mask],
                                          batch['pd_vs_dd'][valid_pd_mask])
                loss_pd = loss_pd + 0.4 * aux_loss_pd  # 40% weight for auxiliary

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
            effective_batch = (batch_idx + 1) // gradient_accumulation_steps

            # Check gradients on first few effective batches (debugging)
            if check_grads and effective_batch <= 3:
                grad_stats = check_gradients(model, log_prefix=f"[Effective Batch {effective_batch}] ")
                print(f"\n[Effective Batch {effective_batch}] Total grad norm: {grad_stats['total_norm']:.4f}, "
                      f"Max: {grad_stats['max_grad']:.4f}, Min: {grad_stats['min_grad']:.4e}")

                # Detailed hierarchical analysis on first effective batch only
                if effective_batch == 1:
                    analyze_hierarchical_gradient_flow(model)

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
            model_outputs = model(batch)
            if len(model_outputs) == 4:  # Auxiliary loss enabled
                logits_hc_vs_pd, logits_pd_vs_dd, _, _ = model_outputs
            else:
                logits_hc_vs_pd, logits_pd_vs_dd = model_outputs

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

        # ============ BALANCED OVERSAMPLING FOR CLASS IMBALANCE ============
        print("\n" + "="*60)
        print("CREATING BALANCED SAMPLER (OVERSAMPLING MINORITY CLASSES)")
        print("="*60)

        # Count class frequencies
        hc_indices = [i for i, pt in enumerate(train_dataset.patient_data) if pt['hc_vs_pd'] == 0]
        pd_indices = [i for i, pt in enumerate(train_dataset.patient_data) if pt['hc_vs_pd'] == 1 and pt['pd_vs_dd'] == 0]
        dd_indices = [i for i, pt in enumerate(train_dataset.patient_data) if pt['pd_vs_dd'] == 1]

        hc_count = len(hc_indices)
        pd_count = len(pd_indices)
        dd_count = len(dd_indices)

        print(f"Original distribution: HC={hc_count}, PD={pd_count}, DD={dd_count}")

        # Calculate weights for each sample (inverse of class frequency)
        # We want to oversample minority classes so they appear as often as majority
        max_count = max(hc_count, pd_count, dd_count)

        sample_weights = torch.zeros(len(train_dataset.patient_data))
        for i, pt in enumerate(train_dataset.patient_data):
            if pt['hc_vs_pd'] == 0:  # HC
                sample_weights[i] = max_count / hc_count
            elif pt['hc_vs_pd'] == 1 and pt['pd_vs_dd'] == 0:  # PD
                sample_weights[i] = max_count / pd_count
            elif pt['pd_vs_dd'] == 1:  # DD
                sample_weights[i] = max_count / dd_count

        # Create weighted sampler
        # This will sample patients with probability proportional to their weight
        # Minority classes get higher weights, so they're sampled more often
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True  # Allow sampling same patient multiple times
        )

        print(f"Sample weights: HC={max_count/hc_count:.2f}x, PD={max_count/pd_count:.2f}x, DD={max_count/dd_count:.2f}x")
        print(f"Expected samples per epoch: ~{max_count} from each class")
        print("="*60 + "\n")
        # ===================================================================

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,  # Use balanced sampler instead of shuffle
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
            seq_len=config['seq_len'],
            use_auxiliary_loss=config.get('use_auxiliary_loss', False)
        ).to(device)

        # Initialize classification head biases to counter class imbalance
        print("\n" + "="*60)
        print("INITIALIZING CLASSIFIER BIASES")
        print("="*60)
        model.initialize_classifier_bias(train_dataset)
        print("="*60 + "\n")

        # Loss functions - NO class weights needed since we balance via oversampling!
        # Keep mild label smoothing to prevent overconfident predictions
        label_smoothing = config.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            print(f"\n[Label Smoothing] Using label_smoothing={label_smoothing}")
        else:
            print(f"\n[Loss] Using standard CrossEntropyLoss (no weights, no smoothing)")

        criterion_hc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_pd = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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

        # Track losses for plotting
        train_losses = []
        val_losses = []

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

                    # Check for num_windows leakage (with differential sampling, some difference is expected)
                    print(f"\n🔍 Checking for num_windows leakage...")
                    print(f"   Note: Differential sampling is enabled (HC=0.70, PD=0.0, DD=0.65)")
                    print(f"   Some window count differences are expected and intentional.")

                    # Collect total windows per patient from validation loader
                    num_windows_by_label_hc = {0: [], 1: []}
                    num_windows_by_label_pd = {0: [], 1: []}

                    for batch in val_loader:
                        # Compute total windows per patient (sum across all tasks)
                        for i in range(len(batch['patient_ids'])):
                            total_windows = batch['window_masks'][i].sum().item()

                            if batch['hc_vs_pd'][i].item() != -1:
                                num_windows_by_label_hc[batch['hc_vs_pd'][i].item()].append(total_windows)
                            if batch['pd_vs_dd'][i].item() != -1:
                                num_windows_by_label_pd[batch['pd_vs_dd'][i].item()].append(total_windows)

                    # Analyze HC vs PD
                    if len(num_windows_by_label_hc[0]) > 0 and len(num_windows_by_label_hc[1]) > 0:
                        avg_hc = np.mean(num_windows_by_label_hc[0])
                        avg_pd = np.mean(num_windows_by_label_hc[1])
                        print(f"   HC vs PD - Avg total windows: HC={avg_hc:.2f}, PD={avg_pd:.2f}")
                        print(f"   (Difference due to differential sampling: HC overlap=0.70, PD overlap=0.0)")

                    # Analyze PD vs DD
                    if len(num_windows_by_label_pd[0]) > 0 and len(num_windows_by_label_pd[1]) > 0:
                        avg_pd2 = np.mean(num_windows_by_label_pd[0])
                        avg_dd = np.mean(num_windows_by_label_pd[1])
                        print(f"   PD vs DD - Avg total windows: PD={avg_pd2:.2f}, DD={avg_dd:.2f}")
                        print(f"   (Difference due to differential sampling: PD overlap=0.0, DD overlap=0.65)")

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

            # Track losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)

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

            # Plot loss curves
            if len(train_losses) > 0 and len(val_losses) > 0:
                plot_loss(train_losses, val_losses, fold_idx=fold_idx+1, output_dir=plot_dir)

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