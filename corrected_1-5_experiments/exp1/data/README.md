# Exp1 Data Schema

This directory contains the data schema and utilities for Experiment 1.

## Files

- **`schema.py`** - Question type inference and candidate answer generation

## Question Types

The schema defines 6 question types for surgical VQA:

| Type | Description | Example Question | Candidates |
|------|-------------|------------------|------------|
| `yes_no` | Binary questions | "Is there a polyp?" | `["yes", "no"]` |
| `color` | Color identification | "What color is the lesion?" | 11 color vocab |
| `size_numeric` | Size/measurement | "What is the size of the polyp?" | None (numeric) |
| `count_numeric` | Counting | "How many instruments are visible?" | None (numeric) |
| `mcq` | Multiple choice | "Choose the correct diagnosis: ..." | From sample |
| `open_ended` | Free-form | "What is the diagnosis?" | None |

## Usage

### Basic Question Type Inference

```python
from schema import infer_question_type

# Example 1: Yes/No question
question = "Is there a polyp?"
qtype = infer_question_type(question)
print(qtype)  # Output: "yes_no"

# Example 2: Color question
question = "What color is the tissue?"
qtype = infer_question_type(question)
print(qtype)  # Output: "color"

# Example 3: Counting question
question = "How many polyps are visible?"
qtype = infer_question_type(question)
print(qtype)  # Output: "count_numeric"
```

### Generate Candidate Answers

```python
from schema import infer_question_type, build_candidates

# Example with yes/no question
question = "Is there bleeding?"
sample = {"question": question, "answer": "yes"}

qtype = infer_question_type(question)
candidates = build_candidates(qtype, sample)
print(candidates)  # Output: ["yes", "no"]

# Example with color question
question = "What color is the polyp?"
sample = {"question": question, "answer": "pink"}

qtype = infer_question_type(question)
candidates = build_candidates(qtype, sample)
print(candidates)  # Output: ["pink", "white", "red", ..., "gray"]

# Example with MCQ
question = "Choose the correct procedure"
sample = {
    "question": question,
    "answer": "colonoscopy",
    "options": ["colonoscopy", "endoscopy", "gastroscopy"]
}

qtype = infer_question_type(question)
candidates = build_candidates(qtype, sample)
print(candidates)  # Output: ["colonoscopy", "endoscopy", "gastroscopy"]
```

### Use in Evaluation Pipeline

```python
import json
from schema import infer_question_type, build_candidates

# Load evaluation results
with open('../results/exp1_evaluation_results.json') as f:
    results = json.load(f)

# Analyze by question type
type_counts = {}
type_accuracy = {}

for pred in results['predictions']:
    # Infer question type
    qtype = infer_question_type(pred['question'])
    
    # Track counts
    if qtype not in type_counts:
        type_counts[qtype] = 0
        type_accuracy[qtype] = []
    
    type_counts[qtype] += 1
    type_accuracy[qtype].append(pred.get('correct', False))

# Print statistics by type
for qtype, count in type_counts.items():
    acc = sum(type_accuracy[qtype]) / len(type_accuracy[qtype]) * 100
    print(f"{qtype:15} - {count:4} questions - {acc:.2f}% accuracy")
```

### Constrained Generation (Advanced)

```python
from schema import infer_question_type, build_candidates

def predict_with_candidates(model, question, image):
    """Predict with constrained output for certain question types."""
    # Infer question type
    qtype = infer_question_type(question)
    
    # Get candidates if applicable
    sample = {"question": question}
    candidates = build_candidates(qtype, sample)
    
    if candidates:
        # Constrained generation: force model to choose from candidates
        prediction = model.generate(
            image=image,
            question=question,
            force_words=candidates,  # hypothetical API
            max_new_tokens=5
        )
    else:
        # Free-form generation
        prediction = model.generate(
            image=image,
            question=question,
            max_new_tokens=50
        )
    
    return prediction, qtype, candidates
```

## Color Vocabulary

The color vocabulary includes 11 common colors in medical imaging:

```python
COLOR_VOCAB = [
    "pink",    # Normal tissue
    "white",   # Artifacts, text
    "red",     # Bleeding, inflammation
    "black",   # Artifacts, dark regions
    "blue",    # Veins, instruments
    "brown",   # Certain lesions
    "yellow",  # Fat, certain pathologies
    "green",   # Bile, certain artifacts
    "purple",  # Certain lesions
    "orange",  # Rare, certain staining
    "gray"     # Instruments, artifacts
]
```

## Statistics from Exp1 Results

Based on Exp1 evaluation (8,984 test samples):

| Question Type | Estimated Count | Notes |
|---------------|----------------|-------|
| `yes_no` | ~4,500 (50%) | Most common type |
| `size_numeric` | ~800 (9%) | Challenging for models |
| `count_numeric` | ~600 (7%) | Moderate difficulty |
| `color` | ~500 (6%) | Well-defined vocabulary |
| `open_ended` | ~2,500 (28%) | Most varied answers |
| `mcq` | ~84 (1%) | Rare in this dataset |

## Extension Ideas

1. **Add size ranges:** Create discrete bins for sizes (e.g., `["<5mm", "5-10mm", "11-20mm", ">20mm"]`)
2. **Add anatomical locations:** Vocabulary for organs (e.g., `["colon", "stomach", "esophagus"]`)
3. **Add instrument types:** Vocabulary for surgical instruments
4. **Add pathology types:** Medical classification vocabularies (Paris classification, etc.)

## See Also

- Parent config: `../config_exp1_actual.yaml`
- Evaluation results: `../../results/exp1_evaluation_results.json`
- Training script: `../../training/train_qwen_lora.py`

