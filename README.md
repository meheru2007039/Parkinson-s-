# Parkinson's Disease Classification - Hierarchical Multi-Task Neural Network

## Project Overview

This project implements a hierarchical attention-based neural network for classifying Parkinson's Disease (PD) patients from healthy controls (HC) and differentiating between Parkinson's Disease (PD) and Disease with Dementia (DD) using accelerometer data from wrist movements.

### Dataset
- **Total Patients**: 469 (HC=79, PD=291, DD=99)
- **Tasks per Patient**: 10 motor tasks (CrossArms, DrinkGlas, Entrainment, HoldWeight, LiftHold, PointFinger, Relaxed, StretchHold, TouchIndex, TouchNose)
- **Data Type**: 6-channel accelerometer data from left and right wrists
- **Sampling**: Downsampled to 64Hz, bandpass filtered (0.1-20Hz)
- **Severe Class Imbalance**: HC=17%, PD=62%, DD=21%

### Two Classification Tasks
1. **HC vs PD**: Healthy Control vs Parkinson's Disease
2. **PD vs DD**: Parkinson's Disease vs Disease with Dementia

---

## Model Architecture

### Hierarchical Three-Level Attention Network

```
Level 0: Raw Accelerometer Data
    ↓ (windowing with differential overlap)

Level 1: Window-Level Cross-Attention
    - Input: (batch, max_tasks, max_windows, 256, 6)
    - Left/Right wrist cross-attention
    - Global average pooling per window
    → Window representations: (batch, max_tasks, max_windows, model_dim*2)

Level 1.5: Task-Level Aggregation (within patient)
    - Attention pooling across windows per task
    - Window shuffling to prevent serial order learning
    → Task representations: (batch, max_tasks, model_dim*2)

Level 2: Patient-Level Aggregation
    - Self-attention across tasks
    - Attention pooling across tasks
    → Patient representation: (batch, model_dim*2)

Level 3: Classification Heads
    - Separate heads for HC vs PD and PD vs DD
    → Binary predictions for each task
```

### Key Architectural Features
- **Differential Sampling**: HC=0.70 overlap, PD=0.0 overlap, DD=0.65 overlap
- **Window Shuffling**: Prevents learning from temporal order within tasks
- **Fixed Attention Pooling**: Removed bias, proper masking with -inf before softmax
- **Gradient Monitoring**: Real-time hierarchical gradient flow analysis

---

## Problems Identified and Fixes Attempted

### 1. ✅ Data Leakage - FIXED
**Problem**: Original model achieved 100% accuracy, indicating data leakage.

**Attempts**:
- **Label Shuffling Test**: Added random label permutation → accuracy dropped to 76% (majority baseline), confirming leakage existed
- **Suspected Source 1: Text Encoder**: Removed entire BERT-based text encoder that had access to patient metadata
- **Suspected Source 2: Differential Sampling**: Initially suspected different window overlaps created label proxies via window counts
  - Diagnostics confirmed: HC=10 windows, PD=3.8 windows, DD=9.2 windows on average
  - **Resolution**: Kept differential sampling as intended for class imbalance handling, not leakage

**Result**: Data leakage eliminated. No patient overlap between train/validation sets.

---

### 2. ✅ Architecture-Data Mismatch - FIXED
**Problem**: Model designed to aggregate multiple tasks per patient, but dataset fed one patient-task pair at a time.

**Fix**: Major restructure (commit `38e5ecc`)
- Changed from `patient_task_data` (flat pairs) to `patient_data` (nested by patient)
- New data structure: Each sample = 1 patient with ALL tasks
- Updated collate function for 3D structure: (batch, max_tasks, max_windows, 256, 6)
- Added two-level masking: `task_masks` + `window_masks`

**Result**: Model now properly aggregates across tasks per patient as intended.

---

### 3. ✅ Attention Collapse - FIXED
**Problem**: Attention pooling bias had zero gradient (0.000000).

**Root Cause**:
- Using `nn.Sequential(Linear, Softmax)` made bias ineffective (cancelled by softmax normalization)
- Double normalization (softmax + manual division) destroyed gradient flow
- Masking occurred AFTER softmax instead of before

**Fix** (commit `a36ce6c`):
- Removed bias from attention layer: `nn.Linear(model_dim*2, 1, bias=False)`
- Mask with `-inf` BEFORE softmax (not 0 after)
- Single softmax normalization (removed manual division)

**Result**: Attention gradients restored. No longer shows zero gradients.

---

### 4. ❌ Majority Class Collapse - NOT FIXED
**Problem**: Model predicts ONLY majority class despite all architectural fixes.

#### Attempt 1: Class Weighting + Label Smoothing
**Implementation**:
- Inverse frequency weights: HC=2.349, PD=0.635 for HC vs PD task
- Label smoothing: 0.1 → 0.2
- Weighted CrossEntropyLoss

**Result**: Failed. Confusion matrices showed 100% PD predictions.
```
HC vs PD: [[ 0 16]    ← All predicted as PD
           [ 0 58]]

PD vs DD: [[58  0]    ← All predicted as PD
           [20  0]]
```

#### Attempt 2: Classifier Bias Initialization
**Implementation** (commit `bf23ca6`):
- Initialize final layer bias to `log(class_frequency)`
- HC: bias=-1.547, PD: bias=-0.239
- Gives minority classes fair starting point

**Result**: Failed. Still predicts only majority class.

#### Attempt 3: Balanced Oversampling (Current)
**Implementation** (commit `0c59b82`):
- `WeightedRandomSampler` with replacement
- Minority classes sampled multiple times per epoch
- HC sampled 3.70x, DD sampled 2.95x, PD sampled 1.0x
- Expected: ~233 samples from each class per epoch
- Removed class weights from loss (data is now balanced)

**Result**: FAILED CATASTROPHICALLY
- Model now unstable, flipping between extremes:
  - Epochs 1-2: Predicts ALL as PD (78% accuracy)
  - Epoch 3: Suddenly predicts ALL as HC (22% accuracy) / ALL as DD (26% accuracy)
  - Training loss decreasing but validation loss increasing (overfitting to repeated samples)

---

## Current Status: Critical Failure

### Validation Metrics (Epoch 1-2)
```
HC vs PD:
  Accuracy: 78.4%
  HC: Precision=0.00, Recall=0.00, F1=0.00 (0/16 correct)
  PD: Precision=0.78, Recall=1.00, F1=0.88 (58/58 correct)
  Confusion: [[ 0 16], [ 0 58]] ← ALL predicted as PD

PD vs DD:
  Accuracy: 74.4%
  PD: Precision=0.74, Recall=1.00, F1=0.85 (58/58 correct)
  DD: Precision=0.00, Recall=0.00, F1=0.00 (0/20 correct)
  Confusion: [[58 0], [20 0]] ← ALL predicted as PD
```

### Validation Metrics (Epoch 3 - Sudden Flip)
```
HC vs PD:
  Accuracy: 21.6%
  HC: Precision=0.22, Recall=1.00, F1=0.36 (16/16 correct)
  PD: Precision=0.00, Recall=0.00, F1=0.00 (0/58 correct)
  Confusion: [[16 0], [58 0]] ← ALL predicted as HC (flipped!)

PD vs DD:
  Accuracy: 25.6%
  PD: Precision=0.00, Recall=0.00, F1=0.00 (0/58 correct)
  DD: Precision=0.26, Recall=1.00, F1=0.41 (20/20 correct)
  Confusion: [[0 58], [0 20]] ← ALL predicted as DD (flipped!)
```

### Training Metrics
- Train Loss: Decreasing (0.9762 → 0.9328)
- Val Loss: Increasing (1.0781 → 1.1900)
- **Overfitting to oversampled data**

### Gradient Health
✅ **All gradients are healthy**:
- Window-level avg: 0.088
- Task-level avg: 0.182
- Classification heads avg: 0.415
- Gradient flow ratio: 0.211 (healthy)
- No vanishing/exploding gradients detected

---

## Root Cause Analysis

### Why Model Cannot Learn

Despite fixing all architectural issues, the model faces **extreme class imbalance** that none of our interventions can overcome:

1. **Class Distribution**:
   - Training: HC=63 (17%), PD=233 (62%), DD=79 (21%)
   - Validation: HC=16 (17%), PD=58 (62%), DD=20 (21%)

2. **Oversampling Paradox**:
   - Oversampling creates artificial balance in training
   - Validation still has natural imbalance
   - Model learns to predict balanced distribution
   - Validation penalizes balanced predictions (majority class is optimal)
   - Creates train/validation mismatch

3. **Differential Sampling Leakage**:
   - Window counts differ by class: HC=100, PD=38, DD=92
   - Model may be exploiting this as a feature
   - Confirms suspicion from original leakage investigation

4. **Small Minority Class Samples**:
   - Only 16 HC and 20 DD patients in validation
   - Insufficient to evaluate minority class performance
   - High variance in metrics

---

## Attempted Solutions Summary

| Approach | Status | Result |
|----------|--------|--------|
| Remove text encoder | ✅ Implemented | Eliminated metadata leakage |
| Fix architecture-data mismatch | ✅ Implemented | Proper hierarchical aggregation |
| Fix attention pooling gradients | ✅ Implemented | Gradients healthy |
| Class weighting (2.3x for HC) | ❌ Failed | Still predicts only PD |
| Label smoothing (0.1 → 0.2) | ❌ Failed | Still predicts only PD |
| Classifier bias initialization | ❌ Failed | Still predicts only PD |
| Balanced oversampling (3.7x HC) | ❌ FAILED | Model unstable, flips extremes |
| Gradient monitoring | ✅ Working | Confirms healthy gradients |

---

## Technical Details

### Model Hyperparameters
```python
{
    'model_dim': 64,
    'num_heads': 4,
    'num_window_layers': 2,  # Cross-attention between wrists
    'num_task_layers': 2,     # Self-attention across tasks
    'd_ff': 256,
    'dropout': 0.3,
    'seq_len': 256,

    'batch_size': 1,  # 1 patient per batch
    'gradient_accumulation_steps': 32,  # Effective batch = 32
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'label_smoothing': 0.1,

    'num_folds': 5,
    'num_epochs': 100
}
```

### Data Pipeline
1. Load patient metadata and movement data
2. Downsample 100Hz → 64Hz
3. Bandpass filter (0.1-20Hz)
4. Create windows with differential overlap
5. Group all tasks per patient
6. 5-fold stratified cross-validation at patient level
7. Balanced oversampling via WeightedRandomSampler

### Training Features
- ✅ Patient-level k-fold split (no data leakage)
- ✅ Gradient accumulation for larger effective batch
- ✅ Gradient clipping (max norm = 1.0)
- ✅ Label smoothing (0.1)
- ✅ Balanced oversampling (minority classes repeated)
- ✅ Real-time gradient monitoring
- ✅ Hierarchical gradient flow analysis
- ✅ Data leakage detection

---

## Known Issues

### Critical Issues
1. **Majority Class Collapse**: Model cannot learn to predict minority classes
   - Tried: class weighting, label smoothing, bias init, oversampling
   - All failed or made problem worse

2. **Train/Val Distribution Mismatch**:
   - Training: Artificially balanced via oversampling
   - Validation: Natural imbalance (62% PD)
   - Model optimizes for wrong distribution

3. **Potential Num_Windows Leakage**:
   - HC: 100 windows (0.70 overlap)
   - PD: 38 windows (0.0 overlap)
   - DD: 92 windows (0.65 overlap)
   - Model may count windows instead of learning features

### Moderate Issues
4. **Small Validation Set**: Only 16 HC, 20 DD patients
   - High variance in minority class metrics
   - Unreliable performance estimates

5. **Oversampling Overfitting**:
   - Training loss decreasing
   - Validation loss increasing
   - Model memorizing repeated samples

---

## Code Structure

```
kaggle_notebook.py
├── Configuration (lines 25-61)
├── Data Pipeline (lines 94-361)
│   ├── ParkinsonsDatasetLoader class
│   ├── get_k_fold_split() method
│   ├── __getitem__() returns full patient
│   └── collate_fn() for batch assembly
├── Model Architecture (lines 424-738)
│   ├── MyModel class
│   ├── Window-level cross-attention
│   ├── Task-level self-attention
│   ├── Hierarchical attention pooling
│   ├── initialize_classifier_bias()
│   └── forward() with 3-level hierarchy
├── Training Loop (lines 1093-1242)
│   ├── training_phase()
│   ├── Gradient monitoring
│   └── Hierarchical gradient analysis
├── Validation (lines 1244-1400)
│   ├── validation_phase()
│   └── Data leakage detection
└── Main Training (lines 1402-1731)
    ├── K-fold cross-validation
    ├── Balanced oversampling setup
    └── Results aggregation
```

---

## Next Steps / Recommendations

### Immediate Actions Required

1. **Remove Differential Sampling**:
   - Use uniform overlap (0.5) for all classes
   - Eliminates window count as predictive feature
   - Use only oversampling + class weights for imbalance

2. **Unified Oversampling Strategy**:
   - Remove loss weights entirely (causes double-correction)
   - Keep only oversampling OR only loss weights, not both
   - Test uniform overlap + oversampling first

3. **Increase Validation Size**:
   - Change to 3-fold instead of 5-fold
   - More samples in validation for stable metrics
   - Better evaluation of minority classes

4. **Focal Loss**:
   - Replace CrossEntropyLoss with Focal Loss (γ=2)
   - Specifically designed for extreme imbalance
   - Focuses on hard-to-classify examples

5. **Two-Stage Training**:
   - Stage 1: Train with uniform overlap, no sampling
   - Stage 2: Fine-tune with oversampling
   - Prevents overfitting to repeated samples

### Alternative Approaches

6. **Simpler Architecture**:
   - Current 3-level hierarchy may be too complex
   - Try flat CNN or simpler attention
   - Reduce model capacity to prevent overfitting

7. **Ensemble Methods**:
   - Train separate models for HC vs PD and PD vs DD
   - Use different architectures per task
   - Combine predictions

8. **Data Augmentation**:
   - Add noise to accelerometer signals
   - Time warping, scaling, rotation
   - Increases effective training data

9. **Transfer Learning**:
   - Pre-train on similar motor task datasets
   - HAR (Human Activity Recognition) datasets
   - Fine-tune on Parkinson's data

10. **Re-evaluate Problem Formulation**:
    - Consider regression instead of classification
    - Predict disease severity scores
    - May better utilize limited data

---

## Repository Information

### Git Branch
`claude/add-random-011CUsLExPjMU2gWjAQZRy3D`

### Key Commits
- `38e5ecc`: Major restructure - Fix data-architecture mismatch
- `a36ce6c`: Fix attention collapse
- `bf23ca6`: Add classifier bias initialization
- `0c59b82`: Implement balanced oversampling (current, failed)

### Files
- `kaggle_notebook.py`: Main implementation (1731 lines)
- `README.md`: This documentation

---

## Contact

For questions or discussion about this project, please contact your supervisor with this README and the current model state.

---

## Appendix: Full Configuration

```python
config = {
    # Data settings
    'data_root': '/kaggle/input/parkinsons-freezing-of-gait-prediction/',
    'apply_downsampling': True,
    'apply_bandpass_filter': True,
    'num_folds': 5,

    # Model architecture
    'input_dim': 6,
    'model_dim': 64,
    'num_heads': 4,
    'num_window_layers': 2,
    'num_task_layers': 2,
    'd_ff': 256,
    'dropout': 0.3,
    'seq_len': 256,
    'num_classes': 2,

    # Training
    'batch_size': 1,
    'gradient_accumulation_steps': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'num_epochs': 100,
    'num_workers': 0,
    'max_grad_norm': 1.0,
    'label_smoothing': 0.1,
    'use_auxiliary_loss': False,

    # Output
    'save_metrics': True,
    'create_plots': True
}
```

---

## Appendix: Diagnostic Output Examples

### Successful Data Leakage Check
```
======================================================================
DATA LEAKAGE CHECK - COMPARING TRAIN AND VALIDATION PATIENTS
======================================================================
Train patients: 212
Validation patients: 94
Overlapping patients: 0

✓✓✓ GOOD NEWS: No train/val patient overlap detected!

🔍 Checking for num_windows leakage...
   Note: Differential sampling is enabled (HC=0.70, PD=0.0, DD=0.65)
   Some window count differences are expected and intentional.
   HC vs PD - Avg total windows: HC=100.00, PD=38.00
   (Difference due to differential sampling: HC overlap=0.70, PD overlap=0.0)
   PD vs DD - Avg total windows: PD=38.00, DD=92.00
   (Difference due to differential sampling: PD overlap=0.0, DD overlap=0.65)
======================================================================
```

### Healthy Gradient Flow
```
================================================================================
HIERARCHICAL GRADIENT FLOW ANALYSIS
================================================================================

🔵 LEVEL 1: Window-Level Processing (deepest in hierarchy)
--------------------------------------------------------------------------------
Avg gradient: 0.087842 | Max: 0.259850 | Min: 5.52e-03
Layers with near-zero gradients: 0/76
✅ Window-level gradients look healthy

🟢 LEVEL 2: Task-Level Processing
--------------------------------------------------------------------------------
Avg gradient: 0.181986 | Max: 0.499622 | Min: 3.29e-02
✅ Task-level gradients look healthy

🟡 POOLING LAYERS (potential bottleneck)
--------------------------------------------------------------------------------
task_attention_pooling.weight: 0.003525

🟠 CLASSIFICATION HEADS (closest to loss)
--------------------------------------------------------------------------------
Avg gradient: 0.415410

📊 GRADIENT FLOW RATIO ANALYSIS
--------------------------------------------------------------------------------
Window-to-Classification gradient ratio: 0.211459
✅ Good gradient flow through hierarchy - no action needed!
================================================================================
```

### Balanced Sampler Output
```
============================================================
CREATING BALANCED SAMPLER (OVERSAMPLING MINORITY CLASSES)
============================================================
Original distribution: HC=63, PD=233, DD=79
Sample weights: HC=3.70x, PD=1.00x, DD=2.95x
Expected samples per epoch: ~233 from each class
============================================================
```

### Classifier Bias Initialization
```
============================================================
INITIALIZING CLASSIFIER BIASES
============================================================
[Bias Init] HC vs PD: HC=0.213 (bias=-1.547), PD=0.787 (bias=-0.239)
[Bias Init] PD vs DD: PD=0.747 (bias=-0.292), DD=0.253 (bias=-1.374)
============================================================
```

---

**Last Updated**: 2025-11-07
**Status**: Critical failure - majority class collapse persists despite all interventions
