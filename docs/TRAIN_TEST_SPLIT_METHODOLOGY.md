# Train/Test Split Methodology

## 📊 Overview

This document describes how the train/test split was created for all experiments in this repository.

---

## 🎯 Split Strategy Summary

**Method**: Random 90/10 split with fixed seed  
**Seed**: 42 (ensures reproducibility)  
**Train**: 90% of data  
**Test**: 10% of data  
**Validation**: Not used (only train/test split)

---

## 📋 Detailed Methodology

### 1. **Data Source**

**Original Dataset**: Kvasir-VQA (Surgical/Endoscopic Visual Question Answering)

The dataset was initially documented as having:
- Total: 5,266 image-question-answer triplets
- Train: 4,213 samples (80%)
- Test: 1,053 samples (20%)

**However**, the actual implementation uses a different split method (see below).

### 2. **Actual Split Implementation**

The code performs a **90/10 random split** at runtime using `numpy`:

```python
# Split into train/val (90/10)
np.random.seed(42)
indices = np.random.permutation(len(self.samples))
split_idx = int(0.9 * len(self.samples))

if split == 'train':
    self.samples = [self.samples[i] for i in indices[:split_idx]]
else:  # test
    self.samples = [self.samples[i] for i in indices[split_idx:]]
```

**Key Characteristics**:
- ✅ **Reproducible**: Fixed seed (42) ensures same split every time
- ✅ **Random Permutation**: Data is shuffled before splitting
- ✅ **Simple**: No stratification by clinical stage or image
- ⚠️ **No Validation Set**: Only train and test splits (no separate validation)

### 3. **Split Applied to Which Data?**

The split is applied **after** stage categorization:

1. **Original Data**: `qa_pairs_train.json` / `qa_pairs_test.json` (from Kvasir)
2. **Stage Categorization**: Questions categorized into Stage 1/2/3 using Qwen2.5-7B-Instruct
3. **Runtime Split**: Each experiment loads data and applies 90/10 split with seed=42

This means:
- The same data file is used for all experiments
- The split happens at runtime, not pre-processed
- Each stage gets its own 90/10 split independently

---

## 🔬 Split Details by Experiment

### Experiment 1: Random Baseline
- **Data**: `qa_pairs_train.json` (all stages mixed)
- **Split**: 90% train, 10% test (seed=42)
- **Total Samples**: ~4,213 QA pairs
- **Test Samples**: ~421 QA pairs

### Experiment 2: Qwen Ordering
- **Data**: `qa_pairs_train.json` (reordered by LLM)
- **Split**: 90% train, 10% test (seed=42)
- **Total Samples**: ~4,213 QA pairs
- **Test Samples**: ~421 QA pairs

### Experiment 3: CXRTrek Sequential
- **Data**: Stage-specific files (`stage1_train.json`, `stage2_train.json`, `stage3_train.json`)
- **Split**: Each stage split independently 90/10 (seed=42)
- **Stage 1**: ~1,600 train → ~1,440 train / ~160 test
- **Stage 2**: ~2,300 train → ~2,070 train / ~230 test
- **Stage 3**: ~300 train → ~270 train / ~30 test

### Experiment 4: Curriculum Learning
- **Data**: Same stage-specific files as CXRTrek Sequential
- **Split**: Identical to CXRTrek Sequential (seed=42)
- Progressive training uses the same train/test samples

---

## 📊 Actual Test Set Statistics

From the evaluation results:

### CXRTrek Sequential Evaluation
```
Total test samples: 4,114 QA pairs
├── Stage 1: 1,586 samples (38.5%)
├── Stage 2: 2,249 samples (54.6%)
└── Stage 3:   279 samples ( 6.9%)
```

### Curriculum Learning Evaluation
```
Total test samples: 4,114 QA pairs
(Same distribution as CXRTrek Sequential)
```

**Note**: The test set has 4,114 samples, which is **significantly larger** than the documented "10% of 4,213 = ~421" samples. This suggests the evaluation uses a different data file or all available data.

---

## ⚠️ Important Observations

### 1. **Documentation vs Implementation Mismatch**

| Aspect            | Documentation          | Implementation      | Reality                |
| ----------------- | ---------------------- | ------------------- | ---------------------- |
| Train Size        | 4,213 (80%)            | 90% of loaded data  | Varies by stage        |
| Test Size         | 1,053 (20%)            | 10% of loaded data  | 4,114 (evaluation)     |
| Validation Set    | Not mentioned          | Not used            | Not used               |
| Split Method      | "80/20 split"          | "90/10 split"       | Varies                 |
| Split Timing      | Pre-processing         | Runtime (seed=42)   | Runtime                |

### 2. **No Stratification**

The split does **not** stratify by:
- Clinical stage (Stage 1/2/3 are split independently)
- Image ID (same image can appear in train and test with different questions)
- Question type
- Answer length

**Implication**: There's potential for **data leakage** if the same image appears in both train and test with similar questions.

### 3. **Test Set Used in Evaluation**

The evaluation scripts load test data differently:

```python
# Use test split (last 10%)
np.random.seed(42)
indices = np.random.permutation(len(self.samples))
split_idx = int(0.9 * len(self.samples))
self.samples = [self.samples[i] for i in indices[split_idx:]]  # Last 10%
```

But the actual test set has 4,114 samples, suggesting:
- **Option A**: A different, larger data file is used for evaluation
- **Option B**: The evaluation uses all available test data (not 10%)
- **Option C**: The original Kvasir test set (1,053 samples) is used separately

---

## 🔍 Verification

To verify the exact split used:

### Check Training Data Size
```bash
# Stage 1
python -c "
import json
data = json.load(open('experiments/cxrtrek_curriculum_learning/data/stage1_train.json'))
print(f'Stage 1 train: {len(data)} samples')
"
```

### Check Test Data Size
```bash
# Check evaluation results
python -c "
import json
results = json.load(open('experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json'))
predictions = results['predictions']
print(f'Total test samples: {len(predictions)}')
"
```

---

## ✅ Recommendations for Reproducibility

### 1. **Document the Actual Split**

The README should state:
```
Train/Test Split:
- Method: Random permutation with seed=42
- Ratio: 90% train, 10% test
- Applied: Runtime (not pre-processed)
- Test Set: 4,114 samples across all stages
```

### 2. **Pre-process the Split**

For better reproducibility, consider:

```python
# scripts/create_fixed_splits.py
import json
import numpy as np

def create_fixed_splits(input_file, output_prefix, seed=42):
    """Create fixed train/test splits."""
    with open(input_file) as f:
        data = json.load(f)
    
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    split_idx = int(0.9 * len(data))
    
    train_data = [data[i] for i in indices[:split_idx]]
    test_data = [data[i] for i in indices[split_idx:]]
    
    with open(f'{output_prefix}_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(f'{output_prefix}_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Run for each stage
create_fixed_splits('stage1_data.json', 'stage1', seed=42)
create_fixed_splits('stage2_data.json', 'stage2', seed=42)
create_fixed_splits('stage3_data.json', 'stage3', seed=42)
```

### 3. **Prevent Image Leakage**

To ensure no image appears in both train and test:

```python
def split_by_images(data, train_ratio=0.9, seed=42):
    """Split by unique images, not QA pairs."""
    # Group QA pairs by image
    image_to_qa = defaultdict(list)
    for item in data:
        image_to_qa[item['image']].append(item)
    
    # Split images
    images = list(image_to_qa.keys())
    np.random.seed(seed)
    np.random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    # Collect QA pairs
    train_data = [qa for img in train_images for qa in image_to_qa[img]]
    test_data = [qa for img in test_images for qa in image_to_qa[img]]
    
    return train_data, test_data
```

---

## 📝 Summary

| Aspect                  | Value                                      |
| ----------------------- | ------------------------------------------ |
| **Split Method**        | Random permutation with fixed seed         |
| **Seed**                | 42                                         |
| **Train Ratio**         | 90%                                        |
| **Test Ratio**          | 10%                                        |
| **Validation Set**      | None                                       |
| **Stratification**      | None                                       |
| **Image Leakage Risk**  | ⚠️ Yes (same image can be in train/test)  |
| **Reproducibility**     | ✅ Yes (fixed seed)                        |
| **Pre-processed Split** | ❌ No (runtime split)                      |
| **Actual Test Size**    | 4,114 samples                              |

---

## 🔗 Code References

### Training Code
- `experiments/cxrtrek_curriculum_learning/scripts/train_progressive_stage.py:73-83`
- Lines implementing 90/10 split with seed=42

### Evaluation Code
- `experiments/cxrtrek_curriculum_learning/scripts/evaluate_curriculum.py:80-84`
- `experiments/cxrtrek_curriculum_learning/scripts/evaluate_cxrtrek_sequential.py:66-70`
- Lines using last 10% for testing

---

## 📧 Questions?

If you have questions about the train/test split:
- Open an issue on GitHub
- GitHub: [@MuhraAlMahri](https://github.com/MuhraAlMahri)

---

**Last Updated**: October 22, 2025

