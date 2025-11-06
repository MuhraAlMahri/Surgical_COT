#!/bin/bash
# Run all Exp1 refactored jobs in sequence
# This script submits jobs with dependencies

echo "========================================================================="
echo "SUBMITTING EXP1 REFACTORED JOBS"
echo "========================================================================="
echo ""

# Create logs directory
mkdir -p "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs"

# Change to slurm directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm"

# Step 1: Sanity overfit (runs first)
echo "Step 1: Submitting sanity overfit test..."
SANITY_JOB=$(sbatch --parsable 01_sanity_overfit.slurm)
echo "  Job ID: $SANITY_JOB"
echo ""

# Step 2: Full training (runs after sanity check completes successfully)
echo "Step 2: Submitting full training (depends on sanity check)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$SANITY_JOB 02_train_exp1.slurm)
echo "  Job ID: $TRAIN_JOB"
echo "  Dependency: afterok:$SANITY_JOB"
echo ""

# Step 3: Prediction (runs after training completes)
echo "Step 3: Submitting prediction generation (depends on training)..."
PREDICT_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB 03_predict_exp1.slurm)
echo "  Job ID: $PREDICT_JOB"
echo "  Dependency: afterok:$TRAIN_JOB"
echo ""

# Step 4: Evaluation (runs after prediction completes)
echo "Step 4: Submitting evaluation (depends on prediction)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$PREDICT_JOB 04_evaluate_exp1.slurm)
echo "  Job ID: $EVAL_JOB"
echo "  Dependency: afterok:$PREDICT_JOB"
echo ""

echo "========================================================================="
echo "ALL JOBS SUBMITTED"
echo "========================================================================="
echo ""
echo "Job Pipeline:"
echo "  1. Sanity Overfit  : $SANITY_JOB (30 min)"
echo "  2. Training        : $TRAIN_JOB (8 hours, after sanity)"
echo "  3. Prediction      : $PREDICT_JOB (2 hours, after training)"
echo "  4. Evaluation      : $EVAL_JOB (15 min, after prediction)"
echo ""
echo "Total estimated time: ~10.75 hours"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo "  squeue -j $SANITY_JOB,$TRAIN_JOB,$PREDICT_JOB,$EVAL_JOB"
echo ""
echo "View logs in:"
echo "  /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/"
echo ""
echo "Cancel all jobs:"
echo "  scancel $SANITY_JOB $TRAIN_JOB $PREDICT_JOB $EVAL_JOB"
echo ""
echo "========================================================================="

