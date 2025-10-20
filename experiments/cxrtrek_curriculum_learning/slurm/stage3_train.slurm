#!/bin/bash
#SBATCH --job-name=curriculum_stage3
#SBATCH --output=../logs/curriculum_stage3_%j.out
#SBATCH --error=../logs/curriculum_stage3_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Stage 3: Clinical Context
# Continues from Stage 2 checkpoint

echo "=============================================="
echo "CURRICULUM LEARNING - STAGE 3 (FINAL)"
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
PREV_CHECKPOINT="$OUTPUT_DIR/stage2_best"

# Training parameters
STAGE=3
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=5e-6

echo ""
echo "Configuration:"
echo "  Stage: $STAGE (Clinical Context)"
echo "  Data: $DATA_PATH"
echo "  Images: $IMAGE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Starting from: $PREV_CHECKPOINT"
echo ""

# Check if Stage 2 checkpoint exists
if [ ! -d "$PREV_CHECKPOINT" ]; then
    echo "ERROR: Stage 2 checkpoint not found at $PREV_CHECKPOINT"
    echo "Please run Stage 2 training first!"
    exit 1
fi

echo "✓ Found Stage 2 checkpoint"
echo ""

# Run training
cd /l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/curriculum_learning

python scripts/train_progressive_stage.py \
    --stage $STAGE \
    --data_path $DATA_PATH \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --prev_checkpoint $PREV_CHECKPOINT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --device cuda

echo ""
echo "=============================================="
echo "STAGE 3 TRAINING COMPLETED"
echo "=============================================="
echo "ALL CURRICULUM LEARNING STAGES COMPLETE!"
echo "End time: $(date)"
echo "=============================================="
echo "Final model: $OUTPUT_DIR/stage3_best"
echo "Next: Run evaluation to compare with baseline"
echo "=============================================="

