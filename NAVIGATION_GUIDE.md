# 🗺️ Navigation Guide - Find Everything Quickly!

**Lost? Start here!** This guide tells you exactly where to find everything in this repository.

---

## 🎯 Quick Links - Most Important Files

| What You Need | Where to Go |
|---------------|-------------|
| **📊 See All Results** | [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) |
| **🏆 Best Model (CXRTrek Sequential)** | [`experiments/cxrtrek_curriculum_learning/`](experiments/cxrtrek_curriculum_learning/) |
| **📈 All Experiments Comparison** | [`docs/COMPLETE_EXPERIMENT_HISTORY.md`](docs/COMPLETE_EXPERIMENT_HISTORY.md) |
| **🔬 Evaluation Results** | [`experiments/cxrtrek_curriculum_learning/evaluation_results/`](experiments/cxrtrek_curriculum_learning/evaluation_results/) |
| **📖 Read First** | [`README.md`](README.md) (this file) |

---

## 📚 Complete Repository Map

```
Surgical_COT/
│
├── 📖 START HERE - Main Documentation
│   ├── README.md                          ⭐ Project overview and quick start
│   ├── NAVIGATION_GUIDE.md                ⭐ THIS FILE - find everything!
│   ├── SETUP_GUIDE.md                     Setup instructions and troubleshooting
│   ├── CONTRIBUTING.md                    How to contribute
│   └── CHANGELOG.md                       Version history
│
├── 📁 docs/ - Detailed Documentation
│   ├── QUICK_REFERENCE_CARD.md            ⭐ One-page summary of all experiments
│   ├── COMPLETE_EXPERIMENT_HISTORY.md     ⭐ Full timeline (Oct 6-18, 2025)
│   ├── VERIFIED_RESULTS_COMPARISON.md     ⭐ CXRTrek vs Curriculum detailed analysis
│   ├── CURRICULUM_VS_CXRTREK_ADVISOR_SUMMARY.md  Advisor-friendly summary
│   └── README.md                          Documentation index
│
├── 🏆 experiments/ - THE MAIN EXPERIMENTS
│   └── cxrtrek_curriculum_learning/       ⭐⭐⭐ ALL 4 EXPERIMENTS ARE HERE!
│       ├── README.md                      Experiment overview
│       ├── scripts/                       Training & evaluation scripts
│       │   ├── train_stage.py             Train individual stages (CXRTrek)
│       │   ├── train_progressive_stage.py Progressive training (Curriculum)
│       │   ├── evaluate_cxrtrek_sequential.py  Evaluate CXRTrek
│       │   └── evaluate_curriculum.py     Evaluate Curriculum
│       ├── slurm/                         SLURM job scripts
│       │   ├── train_stage1.slurm         CXRTrek Stage 1 training
│       │   ├── train_stage2.slurm         CXRTrek Stage 2 training
│       │   ├── train_stage3.slurm         CXRTrek Stage 3 training
│       │   ├── evaluate_cxrtrek_sequential.slurm
│       │   └── evaluate_curriculum.slurm
│       ├── evaluation_results/            ⭐ ALL RESULTS HERE!
│       │   ├── README.md                  Results documentation
│       │   ├── cxrtrek_sequential_evaluation.json  ⭐ 77.59% (WINNER!)
│       │   └── curriculum_results.json    64.24% (catastrophic forgetting)
│       ├── data/                          Training/test data
│       │   ├── qa_pairs_train.json
│       │   ├── qa_pairs_test.json
│       │   ├── stage1_train.json
│       │   ├── stage2_train.json
│       │   └── stage3_train.json
│       └── models/                        Model checkpoints (not in git)
│           ├── stage1_final/
│           ├── stage2_final/
│           └── stage3_final/
│
├── 📊 datasets/ - Raw Datasets
│   ├── Kvasir-VQA/                        Surgical/endoscopy VQA dataset
│   ├── temset/                            TEMSET dataset experiments
│   └── integrated_surgical_dataset/       Combined datasets
│
├── 🔧 Other Directories (older experiments)
│   ├── scripts/                           Utility scripts
│   ├── src/                               Source code utilities
│   ├── training/                          Older training scripts
│   └── evaluation/                        Older evaluation scripts
│
└── 📄 Configuration Files
    ├── requirements.txt                   Python dependencies
    ├── .gitignore                         Git ignore rules
    └── LICENSE                            MIT License
```

---

## 🎓 The 4 Experiments - Where Are They?

All experiments are in: **`experiments/cxrtrek_curriculum_learning/`**

### Experiment 1: Random Ordering Baseline
- **Results**: Mentioned in documentation (64.24% accuracy)
- **Purpose**: Baseline comparison
- **Location**: Results referenced in [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md)

### Experiment 2: Qwen Ordering
- **Results**: 67.12% accuracy (+2.88 vs baseline)
- **Purpose**: LLM-based curriculum
- **Location**: Results referenced in [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md)

### Experiment 3: CXRTrek Sequential ⭐ WINNER
- **Results**: **77.59% accuracy** (+13.35 vs Curriculum)
- **Code**: [`experiments/cxrtrek_curriculum_learning/scripts/train_stage.py`](experiments/cxrtrek_curriculum_learning/scripts/train_stage.py)
- **Evaluation**: [`experiments/cxrtrek_curriculum_learning/scripts/evaluate_cxrtrek_sequential.py`](experiments/cxrtrek_curriculum_learning/scripts/evaluate_cxrtrek_sequential.py)
- **Results File**: [`experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json`](experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json)
- **SLURM Jobs**: [`experiments/cxrtrek_curriculum_learning/slurm/train_stage*.slurm`](experiments/cxrtrek_curriculum_learning/slurm/)

### Experiment 4: Curriculum Learning
- **Results**: 64.24% accuracy (catastrophic forgetting)
- **Code**: [`experiments/cxrtrek_curriculum_learning/scripts/train_progressive_stage.py`](experiments/cxrtrek_curriculum_learning/scripts/train_progressive_stage.py)
- **Evaluation**: [`experiments/cxrtrek_curriculum_learning/scripts/evaluate_curriculum.py`](experiments/cxrtrek_curriculum_learning/scripts/evaluate_curriculum.py)
- **Results File**: [`experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json`](experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json)
- **Architecture**: [`experiments/cxrtrek_curriculum_learning/../curriculum_learning/ARCHITECTURE.md`](curriculum_learning/ARCHITECTURE.md)

---

## 🔍 Common Tasks - Where to Go

### "I want to see the final results"
→ [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) - One page with everything!

### "I want to understand what was done"
→ [`docs/COMPLETE_EXPERIMENT_HISTORY.md`](docs/COMPLETE_EXPERIMENT_HISTORY.md) - Full timeline

### "I want to run the training myself"
→ [`SETUP_GUIDE.md`](SETUP_GUIDE.md) - Installation and setup
→ [`experiments/cxrtrek_curriculum_learning/scripts/`](experiments/cxrtrek_curriculum_learning/scripts/) - Training scripts

### "I want to see the raw results data"
→ [`experiments/cxrtrek_curriculum_learning/evaluation_results/`](experiments/cxrtrek_curriculum_learning/evaluation_results/) - All JSON files

### "I want to reproduce the experiments"
→ [`experiments/cxrtrek_curriculum_learning/slurm/`](experiments/cxrtrek_curriculum_learning/slurm/) - SLURM job templates
→ [`SETUP_GUIDE.md`](SETUP_GUIDE.md) - Step-by-step instructions

### "I want to see the training data"
→ [`experiments/cxrtrek_curriculum_learning/data/`](experiments/cxrtrek_curriculum_learning/data/) - All QA pairs

### "I want to understand the methodology"
→ [`README.md`](README.md) - Overview
→ [`docs/VERIFIED_RESULTS_COMPARISON.md`](docs/VERIFIED_RESULTS_COMPARISON.md) - Detailed analysis
→ [`docs/TRAIN_TEST_SPLIT_METHODOLOGY.md`](docs/TRAIN_TEST_SPLIT_METHODOLOGY.md) - Train/test split details

### "I want to cite this work"
→ [`README.md`](README.md) - Citation section at bottom

---

## 📊 Results Summary (Quick Reference)

| Experiment | Accuracy | Where to Find |
|------------|----------|---------------|
| 🥇 **CXRTrek Sequential** | **77.59%** | [`experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json`](experiments/cxrtrek_curriculum_learning/evaluation_results/cxrtrek_sequential_evaluation.json) |
| 🥈 **Qwen Ordering** | **67.12%** | Documented in [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) |
| 🥉 **Curriculum Learning** | **64.24%** | [`experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json`](experiments/cxrtrek_curriculum_learning/evaluation_results/curriculum_results.json) |
| **Random Baseline** | **64.24%** | Documented in [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) |

---

## 🚀 Getting Started Checklist

- [ ] Read [`README.md`](README.md) for project overview
- [ ] Check [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) for results summary
- [ ] Look at [`experiments/cxrtrek_curriculum_learning/evaluation_results/`](experiments/cxrtrek_curriculum_learning/evaluation_results/) for raw data
- [ ] Read [`SETUP_GUIDE.md`](SETUP_GUIDE.md) if you want to run experiments
- [ ] Check [`docs/COMPLETE_EXPERIMENT_HISTORY.md`](docs/COMPLETE_EXPERIMENT_HISTORY.md) for full details

---

## 🤔 Still Lost?

1. **Start with**: [`README.md`](README.md) - Main overview
2. **Then go to**: [`docs/QUICK_REFERENCE_CARD.md`](docs/QUICK_REFERENCE_CARD.md) - Quick results
3. **For details**: [`docs/COMPLETE_EXPERIMENT_HISTORY.md`](docs/COMPLETE_EXPERIMENT_HISTORY.md) - Everything

**Questions?** Open an issue on GitHub!

---

## 📌 Key Takeaways

✅ **Main experiments**: `experiments/cxrtrek_curriculum_learning/`  
✅ **Best model**: CXRTrek Sequential (77.59%)  
✅ **All results**: `experiments/cxrtrek_curriculum_learning/evaluation_results/`  
✅ **Quick summary**: `docs/QUICK_REFERENCE_CARD.md`  
✅ **Full story**: `docs/COMPLETE_EXPERIMENT_HISTORY.md`  

---

**Last Updated**: October 20, 2025  
**Repository**: https://github.com/MuhraAlMahri/Surgical_COT








