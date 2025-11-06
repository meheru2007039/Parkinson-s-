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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

# ============================================================================
# config
# ============================================================================
def config():
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
        
        'batch_size': 64,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,
        
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


def prepare_text(metadata, questionnaires):
    text_array = []
    
    if metadata:
        text_array.append(f"Age: {metadata.get('age', 'unknown')}")
        text_array.append(f"Gender: {metadata.get('gender', 'unknown')}")
        if metadata.get('age_at_diagnosis'):
            text_array.append(f"Age at diagnosis: {metadata.get('age_at_diagnosis')}")
        if metadata.get('disease_comment'):
            text_array.append(f"Clinical notes: {metadata.get('disease_comment')}")
    
    if questionnaires and 'item' in questionnaires:
        for item in questionnaires['item']:
            q_text = item.get('text', '')
            q_answer = item.get('answer', '')
            if q_text and q_answer:
                text_array.append(f"Q: {q_text} A: {q_answer}")
    
    return " ".join(text_array) if text_array else "No information available."


# ============================================================================
# DataLoader
# ============================================================================
class ParkinsonsDatasetLoader(Dataset):
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 max_windows_per_task: int = 10,  # Max windows to include per task
                 min_windows_per_task: int = 2,   # Min windows required for a task
                 patient_task_data=None,          # For split datasets
                 apply_downsampling=True,
                 apply_bandpass_filter=True, 
                 apply_prepare_text=True):
        
        self.window_size = window_size
        self.max_windows_per_task = max_windows_per_task
        self.min_windows_per_task = min_windows_per_task
        self.apply_downsampling = apply_downsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.apply_prepare_text = apply_prepare_text
        self.data_root = data_root
        
        # Store patient-task combinations
        # Each entry: {patient_id, task_name, left_windows, right_windows, 
        #              hc_vs_pd, pd_vs_dd, patient_text, num_windows}
        self.patient_task_data = []
        
        if data_root is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            self.questionnaires_template = pathlib.Path(data_root) / "questionnaire" / "questionnaire_response_{p:03d}.json"
            
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
            questionnaire_path = pathlib.Path(str(self.questionnaires_template).format(p=patient_id))
            
            if not patient_path.exists():
                continue
            
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', '')
                questionnaire = {}
                try:
                    with open(questionnaire_path, 'r') as f:
                        questionnaire = json.load(f)
                except:
                    pass
                
                # Prepare patient text
                if self.apply_prepare_text:
                    patient_text = prepare_text(metadata, questionnaire)
                else:
                    patient_text = ""
                
                # Determine labels and overlap
                if condition == 'Healthy':
                    hc_vs_pd_label = 0
                    pd_vs_dd_label = -1
                    overlap = 0.70
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1
                    pd_vs_dd_label = 0
                    overlap = 0
                else:
                    hc_vs_pd_label = -1
                    pd_vs_dd_label = 1
                    overlap = 0.65
                
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
                                    'patient_text': patient_text,
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
        patient_text = pt_data['patient_text']
        num_windows = pt_data['num_windows']
        patient_id = pt_data['patient_id']
        task_name = pt_data['task_name']
        
        return {
            'left_windows': left_windows,
            'right_windows': right_windows,
            'hc_vs_pd': hc_vs_pd.squeeze(),
            'pd_vs_dd': pd_vs_dd.squeeze(),
            'patient_text': patient_text,
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
    patient_texts = []
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
        patient_texts.append(item['patient_text'])
        num_windows_list.append(num_win)
        patient_ids.append(item['patient_id'])
        task_names.append(item['task_name'])
    
    return {
        'left_windows': left_windows_padded,      
        'right_windows': right_windows_padded,    
        'masks': masks,                            
        'hc_vs_pd': torch.stack(hc_vs_pd_labels), 
        'pd_vs_dd': torch.stack(pd_vs_dd_labels),
        'patient_texts': patient_texts,
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
from transformers import BertTokenizer, BertModel

class TextTokenizer(nn.Module):
    
    def __init__(self, model_name='bert-base-uncased', output_dim=128, dropout=0.1):
        super().__init__()
        
        self.model_name = model_name
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        input_dim = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, text_list, device):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        tokens = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        if self.training:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        output = outputs.pooler_output
        
        return self.projection(output)

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
        use_text: bool = True,  
        text_encoder_dim: int = 128,  
        fusion_method: str = 'concat',
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.use_text = use_text
        self.fusion_method = fusion_method
        
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
        
        # ========== Text Encoder ==========
        if use_text:
            self.text_encoder = TextTokenizer(output_dim=text_encoder_dim, dropout=dropout)
            
            if fusion_method == 'concat':
                fusion_dim = model_dim * 2 + text_encoder_dim
            elif fusion_method == 'attention':
                fusion_dim = model_dim * 2
                self.fusion_attention = nn.MultiheadAttention(
                    embed_dim=model_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True  
                )
                self.text_to_signal = nn.Linear(text_encoder_dim, model_dim * 2)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            fusion_dim = model_dim * 2

        # ========== Classification Heads ==========
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
        
        # ========== Text Fusion ==========
        if self.use_text and batch['patient_texts'] is not None:
            text_features = self.text_encoder(batch['patient_texts'], device)
            
            if self.fusion_method == 'concat':
                fused_features = torch.cat([task_representation, text_features], dim=1)
            elif self.fusion_method == 'attention':
                text_transformed = self.text_to_signal(text_features).unsqueeze(1)
                signal_features = task_representation.unsqueeze(1)
                
                fused_output, _ = self.fusion_attention(
                    query=signal_features,
                    key=text_transformed,
                    value=text_transformed
                )
                fused_features = fused_output.squeeze(1)
        else:
            fused_features = task_representation
        
        # ========== Classification ==========
        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)
        
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
        
        # Text fusion
        if self.use_text and batch['patient_texts'] is not None:
            text_features = self.text_encoder(batch['patient_texts'], device)
            
            if self.fusion_method == 'concat':
                fused_features = torch.cat([task_representation, text_features], dim=1)
            elif self.fusion_method == 'attention':
                text_transformed = self.text_to_signal(text_features).unsqueeze(1)
                signal_features = task_representation.unsqueeze(1)
                
                fused_output, _ = self.fusion_attention(
                    query=signal_features,
                    key=text_transformed,
                    value=text_transformed
                )
                fused_features = fused_output.squeeze(1)
        else:
            fused_features = task_representation
        
        return {
            'window_features': window_features,  # (batch, max_windows, model_dim*2)
            'task_representation': task_representation,  # (batch, model_dim*2)
            'fused_features': fused_features,  # (batch, fusion_dim)
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
# Trainer
# ============================================================================

def training_phase(model, dataloader, criterion, optimizer, device):
    pass 

def validation_phase(model, dataloader, criterion, device):
    pass

def train_model(config):
    
    dataloader = ParkinsonsDatasetLoader(
        data_root=config['data_root'],
        apply_downsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter'],
        apply_prepare_text=config['apply_prepare_text'],
        split_type=config['split_type'],
        split_ratio=config['split_ratio'],
        train_tasks=config['train_tasks'],
        num_folds=config['num_folds'],
        window_size=256,
        max_windows_per_task=100,
        min_windows_per_task=5
    )
    
    train_set , val_set = dataloader.get_train_val_split()
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(   
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(
        input_dim=config['input_dim'],
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        num_window_layers=config['num_window_layers'],
        num_task_layers=config['num_task_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        seq_len=config['seq_len'],
        use_text=config['use_text']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        train_loss = training_phase(model, train_loader, criterion, optimizer, device)
        #calculate metrics, save csv , plot ruc and tsne 
        val_loss, val_metrics_hc, val_metrics_pd = validation_phase(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

def main():
    config = config()
    
    results = train_model(config)
    
    return results


if __name__ == "__main__":
    results = main()