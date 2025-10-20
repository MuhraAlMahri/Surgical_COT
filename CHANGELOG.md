# Changelog

All notable changes to the Surgical Chain-of-Thought project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-20

### 🎉 Initial Release

Complete implementation of stage-wise medical VQA training strategies.

### Added

#### Core Features
- **CXRTrek Sequential Training**: Three specialized models for clinical reasoning stages
- **Curriculum Learning**: Progressive training through clinical stages
- **Qwen LLM Ordering**: Difficulty-based curriculum using Qwen2.5-7B
- **Random Baseline**: Standard training approach for comparison

#### Models & Training
- Qwen2-VL-2B-Instruct base model integration
- LoRA (Low-Rank Adaptation) fine-tuning implementation
- Multi-stage training pipeline with SLURM job management
- Automatic checkpoint saving and recovery
- Mixed precision training (bfloat16)

#### Evaluation & Analysis
- Comprehensive evaluation scripts for all approaches
- Per-stage accuracy metrics
- Detailed prediction logging (JSON format)
- Catastrophic forgetting analysis
- Cross-experiment comparison tools

#### Data Processing
- Automated question categorization using Qwen2.5-7B-Instruct
- Three-stage clinical reasoning hierarchy:
  - Stage 1: Initial Assessment (quality control, procedure identification)
  - Stage 2: Findings Identification (abnormalities, instruments, landmarks)
  - Stage 3: Clinical Context (diagnosis, reasoning, treatment)
- Train/test split management
- Kvasir-VQA dataset integration

#### Documentation
- Comprehensive README with usage examples
- Complete experiment history (4 experiments, Oct 6-18, 2025)
- Quick reference card for all experiments
- Verified results comparison
- Setup guide with troubleshooting
- Contributing guidelines
- Technical architecture documentation

#### Scripts & Tools
- `categorize_questions_llm.py`: LLM-based question categorization
- `train_stage.py`: Single-stage training script
- `train_progressive_stage.py`: Progressive curriculum training
- `evaluate_cxrtrek_sequential.py`: CXRTrek Sequential evaluation
- `evaluate_curriculum.py`: Curriculum Learning evaluation
- SLURM job submission templates for all experiments

### Results

#### Performance Summary
- **CXRTrek Sequential**: 77.59% overall accuracy ✅ **WINNER**
  - Stage 1: 82.66% (1,311/1,586)
  - Stage 2: 71.90% (1,617/2,249)
  - Stage 3: 94.62% (264/279)
  
- **Qwen Ordering**: 67.12% overall accuracy
  - +2.88 percentage points vs random baseline

- **Curriculum Learning**: 64.24% overall accuracy
  - Stage 1: 41.64% ⚠️ **Catastrophic forgetting detected**
  - Stage 2: 75.12%
  - Stage 3: 99.65%

- **Random Baseline**: 64.24% overall accuracy

#### Key Findings
- Specialized models outperform progressive training by **+13.35 percentage points**
- Progressive curriculum learning suffers from severe catastrophic forgetting
- Stage 3 (Clinical Context) is highly learnable (94-99% accuracy)
- Stage 1 performance critical for overall accuracy (38% of test samples)

### Technical Details

#### Hyperparameters
- Learning Rate: 1e-5
- Batch Size: 2 per device (effective: 16 with gradient accumulation)
- Epochs: 5 for all experiments
- LoRA Rank: 128
- LoRA Alpha: 256
- Optimizer: AdamW
- Scheduler: Linear with warmup (10%)
- Precision: bfloat16

#### Infrastructure
- Hardware: NVIDIA A100 40GB GPUs
- Cluster: SLURM-managed HPC
- Training Time: 8-12 hours per experiment
- Storage: ~100GB for models and data

### Fixed

#### Critical Bugs
1. **Image Processing Error** (Oct 12, 2025)
   - Issue: `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1`
   - Fix: Moved image processing from `__getitem__` to `collate_fn`
   - Impact: Enabled successful Stage 1 training

2. **LoRA Trainability** (Oct 14, 2025)
   - Issue: `ValueError: optimizer got an empty parameter list`
   - Fix: Added `is_trainable=True` to `PeftModel.from_pretrained()`
   - Impact: Enabled progressive training across stages

3. **CUDA Device Error** (Oct 18, 2025)
   - Issue: `RuntimeError: CUDA unknown error`
   - Fix: Added explicit `torch.cuda.set_device(0)` in evaluation scripts
   - Impact: Successful evaluation runs

4. **Data Path Errors** (Oct 18, 2025)
   - Issue: `FileNotFoundError` after directory reorganization
   - Fix: Updated all paths to reflect new structure
   - Impact: Reproducible evaluation results

### Verified

#### Evaluation Jobs
- **Job 147474** (Oct 18, 2025, 22:32-23:08): CXRTrek Sequential
  - 4,114 test samples evaluated
  - Results: 77.59% overall accuracy
  - Output: `cxrtrek_sequential_evaluation.json` (950 KB)

- **Job 147473** (Oct 18, 2025): Curriculum Learning
  - 4,113 test samples evaluated
  - Results: 64.24% overall accuracy
  - Output: `curriculum_results.json` (1.3 MB)

### Known Issues

1. **Catastrophic Forgetting in Curriculum Learning**
   - Progressive training causes Stage 1 accuracy to drop from ~82% to 41%
   - Recommended: Use CXRTrek Sequential approach instead
   - Future: Implement EWC or experience replay

2. **Large Model Checkpoints**
   - Full model checkpoints are 4-5GB each
   - LoRA adapters are ~500MB each
   - Recommendation: Use git-lfs or external storage

3. **Memory Requirements**
   - Minimum 40GB GPU memory required
   - 7B models require 80GB
   - Limited to single-GPU training currently

### Recommendations

#### For Production
✅ **Use CXRTrek Sequential (77.59%)**
- Best overall accuracy
- No catastrophic forgetting
- Modular and maintainable
- Reliable across all question types

#### For Research
- Investigate continual learning techniques for curriculum approach
- Test with larger models (7B, 14B parameters)
- Explore hybrid specialized-progressive approaches
- Add experience replay mechanisms

### Future Work

#### Planned Features (v1.1.0)
- [ ] Multi-GPU training support
- [ ] Model quantization (INT8, INT4)
- [ ] FastAPI deployment example
- [ ] Docker containerization
- [ ] Weights & Biases integration
- [ ] Confidence calibration metrics

#### Research Directions (v2.0.0)
- [ ] Elastic Weight Consolidation (EWC) implementation
- [ ] Experience replay mechanisms
- [ ] Cross-dataset evaluation (SLAKE, VQA-RAD)
- [ ] Attention visualization tools
- [ ] Interactive demo interface

---

## Version History

- **v1.0.0** (2025-10-20): Initial release with 4 complete experiments
  - CXRTrek Sequential: 77.59% ✅
  - Qwen Ordering: 67.12%
  - Curriculum Learning: 64.24%
  - Random Baseline: 64.24%

---

## Contributors

- Lead Researcher: Muhra Al Mahri
- GitHub: [@MuhraAlMahri](https://github.com/MuhraAlMahri)

---

## Acknowledgments

- Qwen Team for Qwen2-VL and Qwen2.5 models
- Kvasir Dataset authors
- Hugging Face for transformers and PEFT libraries
- Research computing cluster support team

---

For detailed information about each experiment, see:
- [Complete Experiment History](docs/COMPLETE_EXPERIMENT_HISTORY.md)
- [Verified Results Comparison](docs/VERIFIED_RESULTS_COMPARISON.md)
- [Quick Reference Card](docs/QUICK_REFERENCE_CARD.md)
