#!/bin/bash
#SBATCH --job-name=curriculum_stage1
#SBATCH --output=../logs/curriculum_stage1_%j.out
#SBATCH --error=../logs/curriculum_stage1_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Stage 1: Initial Assessment
# Trains from base Qwen2-VL-2B-Instruct model

echo "=============================================="
echo "CURRICULUM LEARNING - STAGE 1"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Paths
DATA_PATH="/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/curriculum_learning/data/qwen3_reordered_3stages.json"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/curriculum_learning/checkpoints"

# Training parameters
STAGE=1
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=5e-6

echo ""
echo "Configuration:"
echo "  Stage: $STAGE (Initial Assessment)"
echo "  Data: $DATA_PATH"
echo "  Images: $IMAGE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Starting from: Base Qwen2-VL-2B-Instruct"
echo ""

# Run training
cd /l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/curriculum_learning

python scripts/train_progressive_stage.py \
    --stage $STAGE \
    --data_path $DATA_PATH \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --device cuda

echo ""
echo "=============================================="
echo "STAGE 1 TRAINING COMPLETED"
echo "End time: $(date)"
echo "=============================================="
echo "Checkpoint: $OUTPUT_DIR/stage1_best"
echo "Next: Submit Stage 2 training with this checkpoint"
echo "=============================================="

