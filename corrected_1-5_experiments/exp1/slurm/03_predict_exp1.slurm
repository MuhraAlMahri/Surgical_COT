#!/bin/bash
#SBATCH --job-name=exp1_predict
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/predict_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/predict_%j.err

echo "========================================================================="
echo "EXP1 REFACTORED - PREDICTION GENERATION"
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

# Set HF cache to /tmp
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

echo "Environment setup complete"
echo ""

# Copy model cache to /tmp
echo "Copying Qwen2-VL model cache to /tmp..."
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"

if [ -d "$SOURCE_MODEL_CACHE" ]; then
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✓ Model cache copied successfully"
fi
echo ""

# Run prediction
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "========================================================================="
echo "RUNNING PREDICTION WITH CONSTRAINED DECODING"
echo "========================================================================="
echo "Features:"
echo "  - Constrained decoding for yes/no, color, MCQ"
echo "  - Max 4 tokens for short answers"
echo "  - Type-aware generation"
echo ""

python3 exp1/predict_exp1.py

PREDICT_EXIT_CODE=$?

echo ""
echo "========================================================================="
echo "Prediction completed with exit code: $PREDICT_EXIT_CODE"
echo "========================================================================="

# Check outputs
if [ -f "exp1/outputs/predictions.jsonl" ]; then
    echo ""
    echo "Predictions file created:"
    ls -lh exp1/outputs/predictions.jsonl
    echo ""
    echo "Sample predictions (first 5):"
    head -5 exp1/outputs/predictions.jsonl
    echo ""
    echo "Total predictions:"
    wc -l exp1/outputs/predictions.jsonl
fi

echo ""
echo "========================================================================="
echo "Job completed at: $(date)"
echo "========================================================================="

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

