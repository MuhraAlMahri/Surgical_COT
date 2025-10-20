# Surgical Chain-of-Thought (COT) for Medical Visual Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Investigating stage-wise training strategies for medical VQA using Vision-Language Models**

**🗺️ Lost? See the [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) to find everything quickly!**

This repository contains the complete implementation and results of our comparative study on different training approaches for medical Visual Question Answering (VQA), with a focus on surgical and endoscopic image analysis.

## 🏆 Key Results

| Approach | Overall Accuracy | Best For |
|----------|-----------------|----------|
| **🥇 CXRTrek Sequential** | **77.59%** | Production deployment ✅ |
| **🥈 Qwen Ordering** | **67.12%** | LLM-based curriculum |
| **🥉 Curriculum Learning** | **64.24%** | Research (catastrophic forgetting) |
| Random Baseline | 64.24% | Baseline comparison |

**Winner: CXRTrek Sequential** - Specialized models trained independently per clinical stage, achieving **+13.35 percentage points** over progressive curriculum learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Experiments](#experiments)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

This project investigates whether **stage-wise training strategies** can improve medical VQA performance by organizing questions into a clinical reasoning hierarchy:

1. **Stage 1 - Initial Assessment**: Quality control, procedure identification, artifact detection
2. **Stage 2 - Findings Identification**: Abnormalities, instruments, anatomical landmarks
3. **Stage 3 - Clinical Context**: Diagnosis, clinical reasoning, treatment recommendations

We compare **four different approaches** to organizing and training on these stages using the **Qwen2-VL-2B-Instruct** vision-language model.

---

## 🎯 Research Question

**Does organizing medical VQA training by clinical reasoning stages improve model performance?**

We test this by comparing:
- **Sequential specialized models** (one model per stage)
- **Progressive curriculum learning** (one model trained stage-by-stage)
- **LLM-based curriculum** (questions ordered by Qwen2.5-7B difficulty)
- **Random ordering baseline** (no organization)

---

## 🧪 Experiments

### Experiment 1: Random Ordering Baseline
- **Approach**: Standard training with shuffled questions
- **Result**: **64.24%** overall accuracy
- **Status**: Baseline ✅

### Experiment 2: Qwen LLM Ordering
- **Approach**: Questions ordered by Qwen2.5-7B-Instruct difficulty scores
- **Result**: **67.12%** overall accuracy (+2.88 pts vs baseline)
- **Status**: LLM curriculum helps slightly ✅

### Experiment 3: CXRTrek Sequential (WINNER 🏆)
- **Approach**: Three specialized models, one per stage
  - Stage 1 Model → trained only on Stage 1 questions
  - Stage 2 Model → trained only on Stage 2 questions  
  - Stage 3 Model → trained only on Stage 3 questions
- **Inference**: Route questions to the appropriate specialized model
- **Result**: **77.59%** overall accuracy (+10.47 pts vs Qwen ordering)
- **Per-Stage Performance**:
  - Stage 1: 82.66% (1,311/1,586)
  - Stage 2: 71.90% (1,617/2,249)
  - Stage 3: 94.62% (264/279)
- **Status**: Production-ready ✅

### Experiment 4: Curriculum Learning (Progressive Training)
- **Approach**: Single model trained progressively
  - Train on Stage 1 → save checkpoint
  - Continue training on Stage 2 → save checkpoint
  - Continue training on Stage 3 → final model
- **Result**: **64.24%** overall accuracy (same as random!)
- **Per-Stage Performance**:
  - Stage 1: 41.64% ⚠️ **Severe catastrophic forgetting!**
  - Stage 2: 75.12%
  - Stage 3: 99.65%
- **Status**: Not recommended (catastrophic forgetting) ❌

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
# CUDA 11.8+ (for GPU support)
# 40GB+ GPU memory (for Qwen2-VL-2B-Instruct)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/MuhraAlMahri/Surgical_COT.git
cd Surgical_COT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0.0
transformers>=4.37.0
peft>=0.8.0
qwen-vl-utils
accelerate>=0.26.0
pillow>=10.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 📊 Dataset

We use the **Kvasir-VQA** dataset for surgical/endoscopic visual question answering:

- **Total Samples**: 5,266 image-question-answer triplets
- **Images**: Endoscopic/surgical images from Kvasir dataset
- **Split**: 80% train, 20% test
- **Question Categories**: Automatically categorized into 3 clinical stages using Qwen2.5-7B-Instruct

### Data Structure

```
data/
├── qa_pairs_train.json          # Training questions & answers
├── qa_pairs_test.json           # Test questions & answers
├── stage1_train.json            # Stage 1 training data
├── stage1_test.json             # Stage 1 test data
├── stage2_train.json            # Stage 2 training data
├── stage2_test.json             # Stage 2 test data
├── stage3_train.json            # Stage 3 training data
└── stage3_test.json             # Stage 3 test data

images/
└── [image files from Kvasir dataset]
```

### Stage Categorization (Automated)

Questions are categorized using **Qwen2.5-7B-Instruct** LLM with the following prompt:

```
STAGE 1 - INITIAL ASSESSMENT: Quality control, procedure type, artifacts
STAGE 2 - FINDINGS IDENTIFICATION: Abnormalities, instruments, landmarks
STAGE 3 - CLINICAL CONTEXT: Diagnosis, reasoning, clinical significance
```

**Distribution**:
- Stage 1 (Initial): ~38% of questions (1,586 test samples)
- Stage 2 (Findings): ~55% of questions (2,249 test samples)
- Stage 3 (Clinical): ~7% of questions (279 test samples)

---

## 💻 Usage

### 1. Data Preparation

First, categorize your questions into clinical stages:

```bash
# Categorize questions using Qwen2.5-7B-Instruct LLM
cd experiments/cxrtrek_curriculum_learning
python scripts/categorize_questions_llm.py \
    --input_file data/qa_pairs_train.json \
    --output_dir data/ \
    --model_name "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Training

#### Option A: CXRTrek Sequential (Recommended ✅)

Train three independent specialized models:

```bash
# Stage 1 Model
sbatch experiments/cxrtrek_curriculum_learning/slurm/train_stage1.slurm

# Stage 2 Model
sbatch experiments/cxrtrek_curriculum_learning/slurm/train_stage2.slurm

# Stage 3 Model
sbatch experiments/cxrtrek_curriculum_learning/slurm/train_stage3.slurm
```

#### Option B: Curriculum Learning (Progressive)

Train a single model progressively through all stages:

```bash
cd curriculum_learning
./submit_curriculum_pipeline.sh
```

### 3. Evaluation

#### Evaluate CXRTrek Sequential

```bash
cd experiments/cxrtrek_curriculum_learning
sbatch slurm/evaluate_cxrtrek_sequential.slurm
```

#### Evaluate Curriculum Learning

```bash
cd experiments/cxrtrek_curriculum_learning
sbatch slurm/evaluate_curriculum.slurm
```

### 4. View Results

```bash
# CXRTrek Sequential results
cat experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json

# Curriculum Learning results
cat experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json
```

---

## 📈 Results

### Overall Performance Comparison

| Experiment | Overall Acc | Stage 1 | Stage 2 | Stage 3 | Training Time | Status |
|------------|-------------|---------|---------|---------|---------------|--------|
| **CXRTrek Sequential** | **77.59%** | 82.66% | 71.90% | **94.62%** | 3x independent | ✅ Best |
| Qwen Ordering | 67.12% | - | - | - | Single run | ✅ Good |
| Curriculum Learning | 64.24% | **41.64%** ⚠️ | **75.12%** | **99.65%** | Progressive | ❌ Forgetting |
| Random Baseline | 64.24% | - | - | - | Single run | ✅ Baseline |

### Key Observations

1. **CXRTrek Sequential Wins Overall** (+13.35 pts over Curriculum)
   - Consistent performance across all stages (72-95% range)
   - No catastrophic forgetting
   - Production-ready reliability

2. **Curriculum Learning Shows Severe Forgetting**
   - Stage 1 accuracy drops to 41.64% (vs 82.66% for CXRTrek)
   - Stage 3 reaches 99.65% but at the cost of Stage 1
   - Overall worse than random ordering baseline

3. **Stage 3 (Clinical Reasoning) Is Learnable**
   - Both approaches achieve 94-99% on Stage 3
   - Smallest stage (7% of data) but highest accuracy

4. **Stage 1 Is Critical**
   - Represents 38% of test samples
   - CXRTrek's advantage here (+41 pts) drives overall win

---

## 🔑 Key Findings

### ✅ What Works

1. **Specialized Models > Progressive Training**
   - Training separate models per stage avoids catastrophic forgetting
   - Each model becomes an expert in its clinical reasoning stage

2. **Stage-Based Organization Helps**
   - CXRTrek Sequential (77.59%) >> Random (64.24%)
   - Organizing by clinical reasoning stages improves performance

3. **Clinical Stage 3 Is Highly Learnable**
   - Both approaches achieve >94% on diagnosis/reasoning questions
   - Small dataset (7%) sufficient for high performance

### ❌ What Doesn't Work

1. **Progressive Curriculum Learning**
   - Severe catastrophic forgetting on early stages
   - Stage 1 drops from ~82% to 41% during Stage 2/3 training
   - Overall performance worse than random ordering

2. **Single Model for All Stages**
   - Cannot maintain performance across diverse clinical reasoning types
   - Trade-off between early and late stage performance

### 🎓 Implications

**For Medical VQA Deployment:**
- Use **specialized models** (CXRTrek approach)
- Implement **stage classification** to route questions
- Monitor for **catastrophic forgetting** in continual learning

**For Future Research:**
- Investigate continual learning techniques (EWC, experience replay)
- Test with larger models (7B, 14B parameters)
- Explore hybrid approaches (partial specialization)

---

## 🛠️ Technical Details

### Model Architecture

- **Base Model**: Qwen2-VL-2B-Instruct (2 billion parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - LoRA Rank: 128
  - LoRA Alpha: 256
  - Target Modules: All linear layers in language model
- **Precision**: bfloat16 (mixed precision training)

### Training Hyperparameters

```yaml
Learning Rate: 1e-5
Batch Size: 2 (per device)
Gradient Accumulation: 8 steps (effective batch size: 16)
Epochs: 5 (all experiments)
Optimizer: AdamW
Weight Decay: 0.01
Warmup Ratio: 0.1
LR Scheduler: Linear with warmup
Max Sequence Length: 512 tokens
Image Resolution: Variable (Qwen2-VL native)
```

### Hardware

- **GPU**: NVIDIA A100 40GB (single GPU per job)
- **Cluster**: SLURM-managed HPC cluster
- **Training Time**:
  - Stage 1: ~3-4 hours
  - Stage 2: ~4-5 hours
  - Stage 3: ~1-2 hours

---

## 📂 Repository Structure

```
Surgical_COT/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── experiments/
│   └── cxrtrek_curriculum_learning/
│       ├── README.md                   # Experiment details
│       ├── data/                       # Training/test data
│       │   ├── qa_pairs_train.json
│       │   ├── qa_pairs_test.json
│       │   └── stage{1,2,3}_{train,test}.json
│       ├── images/                     # Medical images
│       ├── scripts/
│       │   ├── categorize_questions_llm.py
│       │   ├── train_stage.py
│       │   ├── evaluate_cxrtrek_sequential.py
│       │   └── evaluate_curriculum.py
│       ├── slurm/
│       │   ├── train_stage{1,2,3}.slurm
│       │   ├── evaluate_cxrtrek_sequential.slurm
│       │   └── evaluate_curriculum.slurm
│       ├── models/
│       │   ├── stage1_final/           # CXRTrek Stage 1 model
│       │   ├── stage2_final/           # CXRTrek Stage 2 model
│       │   └── stage3_final/           # CXRTrek Stage 3 model
│       ├── evaluation_results/
│       │   ├── cxrtrek_sequential_evaluation.json
│       │   └── curriculum_results.json
│       └── logs/                       # Training/evaluation logs
│
├── curriculum_learning/
│   ├── scripts/
│   │   └── train_progressive_stage.py  # Progressive training
│   ├── slurm/
│   │   └── train_stage{1,2,3}.slurm
│   ├── models/
│   │   ├── stage1/                     # After Stage 1 training
│   │   ├── stage2/                     # After Stage 2 training
│   │   └── stage3/                     # Final model
│   └── logs/
│
└── docs/
    ├── COMPLETE_EXPERIMENT_HISTORY.md  # Full experiment timeline
    ├── QUICK_REFERENCE_CARD.md         # One-page summary
    ├── VERIFIED_RESULTS_COMPARISON.md  # Detailed result analysis
    └── CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md
```

---

## 📖 Documentation

- **[Complete Experiment History](docs/COMPLETE_EXPERIMENT_HISTORY.md)** - Detailed timeline of all 4 experiments
- **[Quick Reference Card](docs/QUICK_REFERENCE_CARD.md)** - One-page summary of all experiments
- **[Verified Results Comparison](docs/VERIFIED_RESULTS_COMPARISON.md)** - Detailed analysis of CXRTrek vs Curriculum
- **[Advisor Summary](docs/CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md)** - High-level summary for advisors

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

### Areas for Improvement

1. **Continual Learning Techniques**
   - Implement Elastic Weight Consolidation (EWC)
   - Add experience replay mechanisms
   - Test other regularization approaches

2. **Model Scaling**
   - Test with Qwen2-VL-7B-Instruct
   - Experiment with other VLMs (LLaVA, InstructBLIP)

3. **Dataset Expansion**
   - Add more medical imaging domains
   - Test on other VQA benchmarks

4. **Deployment Optimization**
   - Model quantization (INT8, INT4)
   - Inference optimization
   - API deployment examples

---

## 📄 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{surgical_cot_2025,
  title={Stage-Wise Training Strategies for Medical Visual Question Answering: A Comparative Study},
  author={Muhra Al Mahri},
  journal={[Conference/Journal]},
  year={2025},
  note={Available at: https://github.com/MuhraAlMahri/Surgical_COT}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen2-VL and Qwen2.5 models
- **Kvasir Dataset** authors for the medical image dataset
- **Hugging Face** for transformers and PEFT libraries

---

## 📧 Contact

For questions or collaborations:
- **GitHub**: [@MuhraAlMahri](https://github.com/MuhraAlMahri)
- **GitHub Issues**: [Create an issue](https://github.com/MuhraAlMahri/Surgical_COT/issues)

---

## 🔗 Related Work

- [Qwen2-VL: Vision-Language Models](https://github.com/QwenLM/Qwen2-VL)
- [Kvasir Dataset](https://datasets.simula.no/kvasir/)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for advancing medical AI

</div>
