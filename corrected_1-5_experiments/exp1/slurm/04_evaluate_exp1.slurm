#!/bin/bash
#SBATCH --job-name=exp1_eval
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/eval_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/eval_%j.err

echo "========================================================================="
echo "EXP1 REFACTORED - EVALUATION"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Activate environment
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1

# Run evaluation
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "========================================================================="
echo "RUNNING PER-TYPE EVALUATION"
echo "========================================================================="
echo "Features:"
echo "  - Per-type accuracy breakdown"
echo "  - Numeric tolerance (5% relative)"
echo "  - Normalized string matching"
echo ""

python3 exp1/eval_exp1.py

EVAL_EXIT_CODE=$?

echo ""
echo "========================================================================="
echo "Evaluation completed with exit code: $EVAL_EXIT_CODE"
echo "========================================================================="
echo "Job completed at: $(date)"
echo "========================================================================="

