#!/bin/bash
#SBATCH --job-name=exp1_train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_%j.err

echo "========================================================================="
echo "EXP1 REFACTORED - FULL TRAINING"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Set HF cache to /tmp to avoid quota issues
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

echo "Environment setup complete"
echo "HF_HOME: $HF_HOME"
echo ""

# Copy model cache to /tmp
echo "========================================================================="
echo "STEP 1: Copying Qwen2-VL model cache to /tmp"
echo "========================================================================="
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
DEST_MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct"

if [ -d "$SOURCE_MODEL_CACHE" ]; then
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✓ Model cache copied successfully"
else
    echo "Warning: Source model cache not found at $SOURCE_MODEL_CACHE"
    echo "Model will be downloaded (may take time)"
fi
echo ""

# Cache copied (size check removed to comply with HPC policy)
echo ""

# Run training
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "========================================================================="
echo "STEP 2: RUNNING TRAINING"
echo "========================================================================="
echo "Configuration: exp1/config_exp1.yaml"
echo "Features:"
echo "  - Instruction fine-tuning with label masking"
echo "  - LoRA on LLM only (vision frozen)"
echo "  - Auto-enrichment of JSONL data"
echo "  - Question-type aware prompting"
echo ""

python3 exp1/train_exp1.py exp1/config_exp1.yaml

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "========================================================================="

# Check for outputs
if [ -d "exp1/outputs" ]; then
    echo ""
    echo "Training outputs:"
    ls -lh exp1/outputs/
    echo ""
    
    # Check trainer state
    if [ -f "exp1/outputs/trainer_state.json" ]; then
        echo "Final training state:"
        tail -20 exp1/outputs/trainer_state.json
    fi
fi

echo ""
echo "========================================================================="
echo "Job completed at: $(date)"
echo "========================================================================="

# Cleanup /tmp
echo "Cleaning up /tmp..."
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"
echo "Done"

