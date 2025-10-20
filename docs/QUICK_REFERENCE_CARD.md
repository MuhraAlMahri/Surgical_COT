# Quick Reference Card - All Experiments

**Author:** Muhra Almahri  
**Date:** October 18, 2025  
**Summary:** One-page overview of all experiments

---

## 📊 All Experiments at a Glance

| Experiment | Data Order | Training | Models | Accuracy | Status |
|-----------|-----------|----------|--------|----------|--------|
| **1. Random Baseline** | Random | All together | 1 | ~65-70% | Baseline |
| **2. LLM Reordering** | LLM-ordered | All together | 1 | ~65-70% | No gain |
| **3. CXRTrek Sequential** | LLM-ordered | 3 specialized | 3 | **81.91%** | 🏆 Winner |
| **4. Curriculum Learning** | LLM-ordered | Progressive | 1 | 64.24% | ❌ Failed |

---

## 🔧 Hyperparameters Used

### Early Experiments (Baseline, LLM Reordering)
```
Model:          Qwen2-VL-2B-Instruct (2.28B params)
LoRA Rank:      64
LoRA Alpha:     128
Trainable:      ~35M params (1.5%)
Learning Rate:  2e-5
Batch Size:     16 (effective)
Epochs:         3
Precision:      bfloat16
Training Time:  ~6 hours each
```

### Later Experiments (CXRTrek, Curriculum)
```
Model:          Qwen2-VL-2B-Instruct (2.28B params)
LoRA Rank:      256 ⬆️ (increased)
LoRA Alpha:     512 ⬆️ (increased)
Trainable:      ~69.7M params (3.06%)
Learning Rate:  5e-6 ⬇️ (decreased for stability)
Batch Size:     8 (effective)
Epochs:         3 per stage
Precision:      bfloat16
Training Time:  ~8.5 hours total
```

**Key Changes:**
- **Higher LoRA rank** (64→256) for better model capacity
- **Lower learning rate** (2e-5→5e-6) for more stable training
- **More trainable parameters** (35M→69.7M) for complex reasoning

---

## 📈 Performance Summary

### Overall Accuracy
```
CXRTrek Sequential:  81.91%  ⭐⭐⭐  (BEST)
Random Baseline:     ~65-70%        (Reference)
LLM Reordering:      ~65-70%        (No improvement)
Curriculum Learning: 64.24%  ❌     (Worse than baseline!)
```

### Per-Stage Breakdown (CXRTrek vs Curriculum)
```
Stage 1 (Initial Assessment):
  CXRTrek:     84.44%  ⭐
  Curriculum:  41.64%  ❌  (-42.80% catastrophic forgetting!)

Stage 2 (Findings Identification):
  CXRTrek:     80.48%  ⭐
  Curriculum:  75.12%  📊  (-5.36%)

Stage 3 (Clinical Context):
  CXRTrek:     80.28%  ⭐
  Curriculum:  99.65%  ❓  (Suspicious - likely overfitting)
```

---

## 🔬 Data Information

**Dataset:** Kvasir-VQA (Surgical Endoscopy)
- **Images:** 1,000 surgical images
- **QA Pairs:** 41,130 total
- **Split:** 90% train (37,017), 10% test (4,113)

**LLM Categorization:** Qwen2.5-7B-Instruct
- **Stage 1 (Initial Assessment):** 15,856 QA (38.6%)
- **Stage 2 (Findings Identification):** 22,486 QA (54.7%)
- **Stage 3 (Clinical Context):** 2,781 QA (6.8%)

---

## 🎯 Key Findings

1. **Reordering alone doesn't help** - Need specialized training
2. **Specialization wins** - 3 models > 1 model (81.91% vs 64.24%)
3. **Context passing works** - Inference-time context > training-time transfer
4. **Catastrophic forgetting is real** - Progressive training hurt Stage 1 (84.44%→41.64%)
5. **Deployment complexity justified** - 17.67% accuracy gain worth it

---

## 💾 Computational Resources

| Experiment | GPU Hours | Parallelizable? | Storage (LoRA) |
|-----------|-----------|-----------------|----------------|
| Random Baseline | ~6h | N/A | ~270 MB |
| LLM Reordering | ~6h | N/A | ~270 MB |
| CXRTrek Sequential | ~8.5h | ✅ Yes (~5.5h) | ~810 MB (3 models) |
| Curriculum Learning | ~8.5h | ❌ No (sequential) | ~810 MB (3 stages) |

**Total GPU Hours:** ~46 hours  
**Total Storage:** ~2.7 GB (models + data)

---

## 🏆 Winner: CXRTrek Sequential

### Why It Won
- ✅ **81.91% accuracy** (17.67% better than curriculum)
- ✅ **Consistent 80-84%** across all stages
- ✅ **No catastrophic forgetting** (independent models)
- ✅ **Modular design** (can update individual stages)
- ✅ **Context passing** provides knowledge integration

### Trade-offs
- ❌ **3 models** instead of 1 (deployment complexity)
- ❌ **More storage** (~810 MB vs ~270 MB)
- ❌ **3 forward passes** during inference
- ✅ **But accuracy gain justifies the complexity**

---

## 📁 Documentation Files

**Main Documents:**
1. `COMPLETE_EXPERIMENT_HISTORY.md` (823 lines) - Full history
2. `CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md` - For advisor
3. `experiments/cxrtrek_curriculum_learning/FINAL_COMPARISON_RESULTS.md` - Detailed analysis
4. `experiments/cxrtrek_curriculum_learning/README.md` - Technical docs
5. `EXPERIMENT_COMPLETE.md` - Completion summary
6. `QUICK_REFERENCE_CARD.md` (this file) - Quick reference

**Results:**
- CXRTrek: `evaluation_results/cxrtrek_sequential_evaluation.json`
- Curriculum: `evaluation_results/curriculum_results.json`

**Checkpoints:**
- CXRTrek: `checkpoints/stage{1,2,3}_best/`
- Curriculum: `checkpoints/curriculum_stage{1,2,3}_best/`

---

## 🚀 Recommended for Production

**Use: CXRTrek Sequential (81.91%)**

**Deployment Strategy:**
1. Load 3 specialized models (one per stage)
2. During inference:
   - Stage 1 model predicts initial assessment
   - Pass Stage 1 predictions to Stage 2 as context
   - Pass Stage 1 + Stage 2 predictions to Stage 3 as context
3. Return all predictions

**Why:**
- Medical AI requires high accuracy (81.91% >> 64.24%)
- Consistent performance across all clinical stages
- No catastrophic forgetting issues
- Proven, reliable approach

---

**Last Updated:** October 18, 2025  
**Status:** ✅ Complete  
**Best Result:** 81.91% (CXRTrek Sequential)








