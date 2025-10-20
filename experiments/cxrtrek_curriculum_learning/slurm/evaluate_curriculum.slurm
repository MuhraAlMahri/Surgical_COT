#!/bin/bash
#SBATCH --job-name=eval_curriculum
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/eval_curriculum_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/eval_curriculum_%j.err

echo "=============================================="
echo "CURRICULUM LEARNING EVALUATION"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Paths
CHECKPOINT="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/checkpoints/stage3_best"
DATA="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/data/qwen3_reordered_3stages.json"
IMAGES="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json"
SCRIPT="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/scripts/evaluate_curriculum.py"

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data: $DATA"
echo "  Images: $IMAGES"
echo "  Output: $OUTPUT"
echo ""

# Create output directory
mkdir -p /l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/evaluation_results
mkdir -p /l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs

# Run evaluation
python3 "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --images "$IMAGES" \
    --output "$OUTPUT" \
    --batch_size 4 \
    --device cuda

echo ""
echo "=============================================="
echo "EVALUATION COMPLETED"
echo "=============================================="
echo "Results saved to: $OUTPUT"
echo "End time: $(date)"
echo "=============================================="
