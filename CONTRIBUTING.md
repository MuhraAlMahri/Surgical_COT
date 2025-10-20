# Contributing to Surgical Chain-of-Thought (COT)

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [GitHub Issues](https://github.com/yourusername/Surgical_COT/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU, etc.)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/MuhraAlMahri/Surgical_COT.git
   cd Surgical_COT
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, commented code
   - Follow the existing code style
   - Add tests if applicable

4. **Test your changes**
   ```bash
   # Run existing tests
   pytest tests/
   
   # Test your specific changes
   python scripts/your_script.py
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Comment complex logic

Example:
```python
def evaluate_model(model, test_data, stage_id):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: The trained Qwen2-VL model
        test_data: List of test samples
        stage_id: Clinical stage (1, 2, or 3)
    
    Returns:
        dict: Evaluation metrics (accuracy, predictions)
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md if you add new features
- Add docstrings to new functions
- Create markdown docs for major features
- Include usage examples

### Testing

- Add unit tests for new functions
- Test on different hardware if possible
- Verify on a small dataset first
- Check memory usage for large models

### Commit Messages

Use clear, descriptive commit messages:
- `Add: New feature or file`
- `Fix: Bug fix`
- `Update: Modification to existing code`
- `Docs: Documentation changes`
- `Refactor: Code restructuring`
- `Test: Adding or updating tests`

## 🎯 Areas for Contribution

### High Priority

1. **Continual Learning Improvements**
   - Implement Elastic Weight Consolidation (EWC)
   - Add experience replay mechanisms
   - Test other anti-forgetting techniques

2. **Model Scaling**
   - Test with larger models (Qwen2-VL-7B, 72B)
   - Compare with other VLMs (LLaVA, InstructBLIP)
   - Multi-GPU training support

3. **Evaluation Metrics**
   - Add more medical-specific metrics
   - Implement confidence calibration
   - Error analysis tools

### Medium Priority

4. **Dataset Expansion**
   - Support for other medical VQA datasets
   - Data augmentation techniques
   - Cross-dataset evaluation

5. **Deployment**
   - Model quantization (INT8, INT4)
   - FastAPI deployment example
   - Docker containers

6. **Visualization**
   - Attention visualization
   - Error analysis dashboards
   - Training curves and comparisons

### Nice to Have

7. **Documentation**
   - Video tutorials
   - Jupyter notebook examples
   - Interactive demos

8. **Testing**
   - Unit test coverage
   - Integration tests
   - Continuous integration setup

## 🐛 Bug Reports

When reporting bugs, include:

1. **Environment**
   - OS and version
   - Python version
   - PyTorch version
   - GPU type and memory
   - CUDA version

2. **Steps to Reproduce**
   ```bash
   # Example
   cd experiments/cxrtrek_curriculum_learning
   python scripts/train_stage.py --stage 1
   # Error occurs here
   ```

3. **Error Messages**
   ```
   Full error traceback here
   ```

4. **Expected Behavior**
   What should have happened?

5. **Actual Behavior**
   What actually happened?

## 💡 Feature Requests

For feature requests, describe:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Have you considered other approaches?
4. **Additional Context**: Any other relevant information?

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## ✅ Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] Large files are not committed (use .gitignore)

## 🙏 Recognition

Contributors will be:
- Listed in the project README
- Acknowledged in publications (for major contributions)
- Added to the CONTRIBUTORS.md file

## 📧 Questions?

If you have questions:
- Open a [Discussion](https://github.com/MuhraAlMahri/Surgical_COT/discussions)
- GitHub: [@MuhraAlMahri](https://github.com/MuhraAlMahri)
- Check existing documentation in `/docs`

---

Thank you for contributing to advancing medical AI! 🏥🤖

