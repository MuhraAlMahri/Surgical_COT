#!/bin/bash
#SBATCH --job-name=exp1_sanity
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/sanity_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/sanity_%j.err

echo "========================================================================="
echo "EXP1 REFACTORED - SANITY OVERFIT TEST"
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
echo "Copying Qwen2-VL model cache to /tmp..."
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
DEST_MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct"

if [ -d "$SOURCE_MODEL_CACHE" ]; then
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✓ Model cache copied successfully"
else
    echo "Warning: Source model cache not found, will download"
fi
echo ""

# Run sanity overfit test
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "========================================================================="
echo "RUNNING SANITY OVERFIT TEST"
echo "========================================================================="
echo "Expected: Loss should drop below 2.0 within 200 steps"
echo "If loss stays above 7.0, there's a label masking bug"
echo ""

python3 exp1/sanity_overfit.py

echo ""
echo "========================================================================="
echo "Job completed at: $(date)"
echo "========================================================================="

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

