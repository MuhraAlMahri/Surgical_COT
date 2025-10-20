# Setup Guide for Surgical COT

Complete guide for setting up the Surgical Chain-of-Thought environment and running experiments.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Quick Start](#quick-start)
5. [Running Experiments](#running-experiments)
6. [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Hardware

**Minimum Requirements:**
- GPU: NVIDIA GPU with 40GB VRAM (e.g., A100 40GB)
- RAM: 32GB system memory
- Storage: 100GB free space (for models, data, and checkpoints)
- CPU: 8+ cores recommended

**Recommended:**
- GPU: NVIDIA A100 80GB or multiple A100 40GB
- RAM: 64GB+ system memory
- Storage: 500GB+ SSD
- CPU: 16+ cores

### Software

- **OS**: Linux (Ubuntu 20.04+, CentOS 7+, or similar)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher
- **CUDA Toolkit**: Matching your CUDA version
- **SLURM**: For cluster job management (optional but recommended)

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/MuhraAlMahri/Surgical_COT.git
cd Surgical_COT
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# OR using conda
conda create -n surgical_cot python=3.10
conda activate surgical_cot
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (choose your CUDA version)
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
CUDA Version: 11.8 (or your version)
Transformers: 4.37.x
```

---

## 📊 Data Setup

### Step 1: Download Kvasir Dataset

```bash
# Create data directory
mkdir -p experiments/cxrtrek_curriculum_learning/data
mkdir -p experiments/cxrtrek_curriculum_learning/images

# Download Kvasir-VQA dataset
# (Replace with actual download link or instructions)
cd experiments/cxrtrek_curriculum_learning/data
wget [KVASIR_VQA_URL] -O kvasir_vqa.zip
unzip kvasir_vqa.zip
```

### Step 2: Prepare Data Format

Your data should be in JSON format:

**qa_pairs_train.json:**
```json
[
  {
    "image": "image_001.jpg",
    "question": "What type of procedure is shown?",
    "answer": "Colonoscopy"
  },
  ...
]
```

**qa_pairs_test.json:** (same format)

### Step 3: Categorize Questions into Stages

```bash
cd experiments/cxrtrek_curriculum_learning

# Run LLM categorization
python scripts/categorize_questions_llm.py \
    --input_file data/qa_pairs_train.json \
    --output_dir data/ \
    --model_name "Qwen/Qwen2.5-7B-Instruct"
```

This will create:
- `stage1_train.json`, `stage1_test.json`
- `stage2_train.json`, `stage2_test.json`
- `stage3_train.json`, `stage3_test.json`

### Step 4: Verify Data

```bash
# Check data structure
python scripts/verify_data.py

# Expected output:
# ✓ Total train samples: 4,114
# ✓ Total test samples: 1,152
# ✓ Stage 1: 1,586 test samples
# ✓ Stage 2: 2,249 test samples
# ✓ Stage 3: 279 test samples
```

---

## 🎯 Quick Start

### Option 1: Run Pre-trained Evaluation (Fastest)

If you have access to our pre-trained models:

```bash
cd experiments/cxrtrek_curriculum_learning

# Evaluate CXRTrek Sequential
sbatch slurm/evaluate_cxrtrek_sequential.slurm

# Check results
cat evaluation_results/cxrtrek_sequential_evaluation.json
```

### Option 2: Train from Scratch

**Training CXRTrek Sequential (Recommended):**

```bash
cd experiments/cxrtrek_curriculum_learning

# Submit all three stage training jobs
sbatch slurm/train_stage1.slurm
sbatch slurm/train_stage2.slurm
sbatch slurm/train_stage3.slurm

# Monitor progress
squeue -u $USER
tail -f logs/train_stage1_*.out
```

**Training Curriculum Learning (Experimental):**

```bash
cd curriculum_learning

# Submit progressive training pipeline
./submit_curriculum_pipeline.sh

# Monitor progress
tail -f logs/train_stage1_*.out
```

---

## 🧪 Running Experiments

### Experiment 1: Random Baseline

```bash
cd experiments/random_baseline

# Train model with random ordering
sbatch slurm/train_random.slurm

# Evaluate
sbatch slurm/evaluate_random.slurm

# View results
cat evaluation_results/random_results.json
```

### Experiment 2: Qwen LLM Ordering

```bash
cd experiments/qwen_ordering

# Order questions by difficulty using Qwen2.5-7B
python scripts/order_by_difficulty.py

# Train on ordered data
sbatch slurm/train_qwen_ordered.slurm

# Evaluate
sbatch slurm/evaluate_qwen_ordered.slurm
```

### Experiment 3: CXRTrek Sequential (Recommended)

```bash
cd experiments/cxrtrek_curriculum_learning

# Train specialized models
sbatch slurm/train_stage1.slurm  # ~3-4 hours
sbatch slurm/train_stage2.slurm  # ~4-5 hours
sbatch slurm/train_stage3.slurm  # ~1-2 hours

# Evaluate all stages
sbatch slurm/evaluate_cxrtrek_sequential.slurm

# View results
cat evaluation_results/cxrtrek_sequential_evaluation.json
```

### Experiment 4: Curriculum Learning

```bash
cd curriculum_learning

# Submit progressive training (Stage 1 → 2 → 3)
./submit_curriculum_pipeline.sh

# This will:
# 1. Train Stage 1 (3-4 hours)
# 2. Continue training with Stage 2 (4-5 hours)
# 3. Continue training with Stage 3 (1-2 hours)

# Evaluate final model
cd ../experiments/cxrtrek_curriculum_learning
sbatch slurm/evaluate_curriculum.slurm

# View results
cat evaluation_results/curriculum_results.json
```

---

## 🔧 Configuration

### Modifying Hyperparameters

Edit the training scripts or SLURM files:

```bash
# Example: experiments/cxrtrek_curriculum_learning/slurm/train_stage1.slurm

python scripts/train_stage.py \
    --stage 1 \
    --learning_rate 1e-5 \        # Adjust learning rate
    --num_epochs 5 \               # Change number of epochs
    --batch_size 2 \               # Adjust batch size
    --lora_rank 128 \              # Change LoRA rank
    --lora_alpha 256               # Change LoRA alpha
```

### Using Different Models

To use a different base model:

```python
# In scripts/train_stage.py

# Change this line:
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",  # Replace with your model
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Supported models:
- `Qwen/Qwen2-VL-2B-Instruct` (default, 40GB VRAM)
- `Qwen/Qwen2-VL-7B-Instruct` (80GB VRAM)
- Other vision-language models (may require code modifications)

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in training script
   ```bash
   --batch_size 1  # Instead of 2
   ```

2. Increase gradient accumulation
   ```bash
   --gradient_accumulation_steps 16  # Instead of 8
   ```

3. Use smaller model
   ```bash
   --model_name "Qwen/Qwen2-VL-2B-Instruct"  # Instead of 7B
   ```

### Issue 2: Import Errors

**Error:** `ModuleNotFoundError: No module named 'qwen_vl_utils'`

**Solution:**
```bash
pip install qwen-vl-utils
# OR
pip install -r requirements.txt --upgrade
```

### Issue 3: SLURM Job Fails

**Error:** Job fails immediately after submission

**Solutions:**
1. Check SLURM output
   ```bash
   cat logs/train_stage1_[JOBID].out
   ```

2. Verify GPU availability
   ```bash
   sinfo -p gpu  # Check GPU partition
   ```

3. Check resource requests
   ```bash
   # In SLURM file, adjust:
   #SBATCH --gres=gpu:1  # Request 1 GPU
   #SBATCH --mem=64G     # Adjust memory
   ```

### Issue 4: Slow Training

**Symptoms:** Training is very slow (>10 hours for Stage 1)

**Solutions:**
1. Enable mixed precision (already default in our scripts)
2. Increase batch size if memory allows
3. Check GPU utilization:
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```
4. Ensure data is on fast storage (not network drive)

### Issue 5: Catastrophic Forgetting

**Symptoms:** Stage 1 accuracy drops during curriculum learning

**Solutions:**
1. Use CXRTrek Sequential instead (specialized models)
2. Implement Elastic Weight Consolidation (EWC)
3. Add experience replay
4. Reduce learning rate for later stages

---

## 📊 Monitoring Training

### Using SLURM

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/train_stage1_[JOBID].out

# Check job details
scontrol show job [JOBID]

# View job history
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Using Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
# View at https://wandb.ai/your-username/surgical-cot
```

### Manual Monitoring

```bash
# Check GPU usage
nvidia-smi

# Monitor training loss
grep "train_loss" logs/train_stage1_*.out

# Check checkpoint creation
ls -lh experiments/cxrtrek_curriculum_learning/models/stage1_final/
```

---

## 📁 Expected Directory Structure After Setup

```
Surgical_COT/
├── experiments/
│   └── cxrtrek_curriculum_learning/
│       ├── data/
│       │   ├── qa_pairs_train.json     ✓
│       │   ├── qa_pairs_test.json      ✓
│       │   ├── stage1_train.json       ✓
│       │   ├── stage2_train.json       ✓
│       │   └── stage3_train.json       ✓
│       ├── images/
│       │   └── [image files]           ✓
│       └── models/
│           ├── stage1_final/           (after training)
│           ├── stage2_final/           (after training)
│           └── stage3_final/           (after training)
└── venv/                               ✓
```

---

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] Python 3.8+ installed
- [ ] CUDA available (`torch.cuda.is_available()` returns `True`)
- [ ] All requirements installed (`pip list | grep torch`)
- [ ] Data files present in `experiments/cxrtrek_curriculum_learning/data/`
- [ ] Images present in `experiments/cxrtrek_curriculum_learning/images/`
- [ ] Stage-wise data created (stage1/2/3_train.json and test.json)
- [ ] SLURM working (if using cluster)
- [ ] GPU memory sufficient (40GB+ for Qwen2-VL-2B)

---

## 🎓 Next Steps

Once setup is complete:

1. **Start with Quick Evaluation** (if pre-trained models available)
   ```bash
   cd experiments/cxrtrek_curriculum_learning
   sbatch slurm/evaluate_cxrtrek_sequential.slurm
   ```

2. **Or Train from Scratch**
   ```bash
   sbatch slurm/train_stage1.slurm
   ```

3. **Read the Documentation**
   - [Complete Experiment History](docs/COMPLETE_EXPERIMENT_HISTORY.md)
   - [Quick Reference Card](docs/QUICK_REFERENCE_CARD.md)

4. **Join the Community**
   - Open issues for questions
   - Submit PRs for improvements
   - Share your results!

---

## 📧 Support

If you encounter issues:
1. Check [Troubleshooting](#troubleshooting) section above
2. Search [GitHub Issues](https://github.com/MuhraAlMahri/Surgical_COT/issues)
3. Open a new issue with details
4. GitHub: [@MuhraAlMahri](https://github.com/MuhraAlMahri)

---

**Happy Experimenting! 🚀**

