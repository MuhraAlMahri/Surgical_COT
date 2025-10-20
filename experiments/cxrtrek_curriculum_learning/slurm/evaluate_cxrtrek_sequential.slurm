#!/bin/bash
#SBATCH --job-name=eval_cxrtrek
#SBATCH --output=../logs/eval_cxrtrek_%j.out
#SBATCH --error=../logs/eval_cxrtrek_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00

# CXRTrek Sequential Evaluation Job
# Evaluates three specialized models on their respective test data

echo "=========================================="
echo "CXRTrek Sequential Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
DATA_PATH="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/data/qwen3_reordered_3stages.json"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
STAGE1_CHECKPOINT="${BASE_DIR}/checkpoints/stage1_best"
STAGE2_CHECKPOINT="${BASE_DIR}/checkpoints/stage2_best"
STAGE3_CHECKPOINT="${BASE_DIR}/checkpoints/stage3_best"
OUTPUT_FILE="${BASE_DIR}/evaluation_results/cxrtrek_sequential_evaluation.json"

echo "Data path: $DATA_PATH"
echo "Image directory: $IMAGE_DIR"
echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Stage 2 checkpoint: $STAGE2_CHECKPOINT"
echo "Stage 3 checkpoint: $STAGE3_CHECKPOINT"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="

# Check if checkpoints exist
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    exit 1
fi

if [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "ERROR: Stage 2 checkpoint not found at $STAGE2_CHECKPOINT"
    exit 1
fi

if [ ! -d "$STAGE3_CHECKPOINT" ]; then
    echo "ERROR: Stage 3 checkpoint not found at $STAGE3_CHECKPOINT"
    exit 1
fi

echo "All checkpoints found!"
echo "=========================================="

# Run evaluation
python ${BASE_DIR}/scripts/evaluate_cxrtrek_sequential.py \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --stage2_checkpoint "$STAGE2_CHECKPOINT" \
    --stage3_checkpoint "$STAGE3_CHECKPOINT" \
    --output_file "$OUTPUT_FILE" \
    --device cuda \
    --batch_size 1

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
