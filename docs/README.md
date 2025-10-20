# Documentation

Complete documentation for the Surgical Chain-of-Thought (COT) project.

## 📚 Documentation Index

### Getting Started
- **[Main README](../README.md)** - Project overview, installation, and quick start
- **[Setup Guide](../SETUP_GUIDE.md)** - Detailed setup instructions and troubleshooting
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute to the project

### Research & Results
- **[Complete Experiment History](COMPLETE_EXPERIMENT_HISTORY.md)** - Full timeline of all 4 experiments (Oct 6-18, 2025)
- **[Verified Results Comparison](VERIFIED_RESULTS_COMPARISON.md)** - Detailed analysis of CXRTrek vs Curriculum Learning
- **[Quick Reference Card](QUICK_REFERENCE_CARD.md)** - One-page summary of all experiments
- **[Advisor Summary](CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md)** - Executive summary for advisors

### Technical Documentation
- **[Architecture Guide](../curriculum_learning/ARCHITECTURE.md)** - System architecture and design
- **[Hyperparameters](../curriculum_learning/HYPERPARAMETERS.md)** - Complete hyperparameter documentation
- **[Bugfix Log](../curriculum_learning/BUGFIX_LOG.md)** - Technical fixes applied during development

### Experiment-Specific
- **[CXRTrek Sequential README](../experiments/cxrtrek_curriculum_learning/README.md)** - CXRTrek experiment details
- **[Training Results](../curriculum_learning/TRAINING_RESULTS.md)** - Progressive training results
- **[Training Status](../curriculum_learning/TRAINING_STATUS.md)** - Real-time training progress

## 📊 Key Results Summary

| Approach | Overall Accuracy | Status |
|----------|-----------------|--------|
| 🥇 CXRTrek Sequential | **77.59%** | ✅ Production-Ready |
| 🥈 Qwen Ordering | 67.12% | ✅ Good |
| 🥉 Curriculum Learning | 64.24% | ⚠️ Catastrophic Forgetting |
| Random Baseline | 64.24% | ✅ Baseline |

**Winner: CXRTrek Sequential** (+13.35 percentage points over Curriculum Learning)

## 🎯 Research Question

**Does organizing medical VQA training by clinical reasoning stages improve model performance?**

**Answer: YES** - But only with specialized models (CXRTrek Sequential), not progressive training (Curriculum Learning).

## 📖 Reading Guide

### For Researchers
1. Start with [Complete Experiment History](COMPLETE_EXPERIMENT_HISTORY.md)
2. Review [Verified Results Comparison](VERIFIED_RESULTS_COMPARISON.md)
3. Check technical details in experiment-specific READMEs

### For Practitioners
1. Read [Main README](../README.md) for quick overview
2. Follow [Setup Guide](../SETUP_GUIDE.md) to get started
3. Use [Quick Reference Card](QUICK_REFERENCE_CARD.md) for hyperparameters

### For Advisors/PIs
1. Read [Advisor Summary](CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md)
2. Check [Quick Reference Card](QUICK_REFERENCE_CARD.md) for key results
3. Review methodology in [Complete Experiment History](COMPLETE_EXPERIMENT_HISTORY.md)

### For Contributors
1. Read [Contributing Guide](../CONTRIBUTING.md)
2. Review [Architecture Guide](../curriculum_learning/ARCHITECTURE.md)
3. Check [Bugfix Log](../curriculum_learning/BUGFIX_LOG.md) for known issues

## 🔍 Quick Links

### Results & Data
- [CXRTrek Sequential Evaluation Results](../experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json) (950 KB, Job 147474)
- [Curriculum Learning Evaluation Results](../experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json) (1.3 MB, Job 147473)

### Code
- [CXRTrek Training Script](../experiments/cxrtrek_curriculum_learning/scripts/train_stage.py)
- [Curriculum Training Script](../curriculum_learning/scripts/train_progressive_stage.py)
- [CXRTrek Evaluation Script](../experiments/cxrtrek_curriculum_learning/scripts/evaluate_cxrtrek_sequential.py)
- [Curriculum Evaluation Script](../experiments/cxrtrek_curriculum_learning/scripts/evaluate_curriculum.py)

### SLURM Jobs
- [CXRTrek Training Jobs](../experiments/cxrtrek_curriculum_learning/slurm/)
- [Curriculum Training Jobs](../curriculum_learning/slurm/)

## 🏆 Key Findings

### ✅ What Works
1. **Specialized Models (CXRTrek Sequential)**: 77.59% accuracy
   - One model per clinical stage
   - No catastrophic forgetting
   - Production-ready reliability

2. **Stage-Based Organization**: Significant improvement over random
   - Clinical reasoning hierarchy works
   - Stage 3 (Clinical Context) highly learnable (94-99%)

### ❌ What Doesn't Work
1. **Progressive Training (Curriculum Learning)**: 64.24% accuracy
   - Severe catastrophic forgetting on Stage 1 (41.64%)
   - Not suitable for deployment
   - Needs continual learning techniques

2. **Single Model for All Stages**
   - Cannot maintain performance across diverse reasoning types

## 📈 Performance Breakdown

### CXRTrek Sequential (77.59%) ✅
```
Stage 1 (Initial Assessment):    82.66% (1,311/1,586 samples)
Stage 2 (Findings):               71.90% (1,617/2,249 samples)
Stage 3 (Clinical Context):       94.62% (264/279 samples)
```

### Curriculum Learning (64.24%) ❌
```
Stage 1 (Initial Assessment):    41.64% (645/1,550 samples) ⚠️ FORGETTING
Stage 2 (Findings):               75.12% (1,708/2,274 samples)
Stage 3 (Clinical Context):       99.65% (288/289 samples)
```

## 🔬 Methodology

### Clinical Stage Hierarchy
1. **Stage 1 - Initial Assessment** (38% of data)
   - Quality control
   - Procedure identification
   - Artifact detection

2. **Stage 2 - Findings Identification** (55% of data)
   - Abnormalities
   - Instruments
   - Anatomical landmarks

3. **Stage 3 - Clinical Context** (7% of data)
   - Diagnosis
   - Clinical reasoning
   - Treatment recommendations

### Stage Categorization
Questions categorized using **Qwen2.5-7B-Instruct** LLM with structured prompt.

### Model
- **Base**: Qwen2-VL-2B-Instruct (2B parameters)
- **Fine-tuning**: LoRA (rank 128, alpha 256)
- **Hardware**: NVIDIA A100 40GB
- **Training**: 5 epochs, learning rate 1e-5

## 🎓 Publications & Citations

### How to Cite

```bibtex
@article{surgical_cot_2025,
  title={Stage-Wise Training Strategies for Medical Visual Question Answering: A Comparative Study},
  author={Muhra Al Mahri},
  journal={[Conference/Journal]},
  year={2025},
  note={Code available at: https://github.com/MuhraAlMahri/Surgical_COT}
}
```

## 🤝 Support & Contact

- **GitHub Issues**: [Create an issue](https://github.com/MuhraAlMahri/Surgical_COT/issues)
- **GitHub**: [@MuhraAlMahri](https://github.com/MuhraAlMahri)
- **Documentation**: This directory

## 📝 Documentation Standards

When adding new documentation:
1. Use Markdown (.md) format
2. Include table of contents for long documents
3. Add code examples where applicable
4. Link to related documentation
5. Update this index file

## 🔄 Updates

Documentation is regularly updated to reflect:
- New experimental results
- Bug fixes and improvements
- Community contributions
- Best practices and recommendations

Last updated: October 20, 2025

---

**Need help?** Start with the [Main README](../README.md) or [Setup Guide](../SETUP_GUIDE.md).

