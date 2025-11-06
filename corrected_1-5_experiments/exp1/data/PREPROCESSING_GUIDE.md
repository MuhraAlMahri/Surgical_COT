# Data Preprocessing Guide

## Overview

The preprocessing utilities enrich your surgical VQA data with:
- **Question type inference** (automatic categorization)
- **Answer normalization** (consistent formatting)
- **Answer candidates** (valid options for constrained generation)

---

## Files

### 1. `preprocess.py` - Core Functions
```python
from exp1.data import normalize_answer, enrich_jsonl

# Normalize an answer
normalized = normalize_answer("Yes, there is.")  # → "yes there is"

# Enrich a JSONL file
enrich_jsonl("input.jsonl", "output_enriched.jsonl")
```

### 2. `preprocess_cli.py` - Command-Line Tool
```bash
python preprocess_cli.py input.jsonl output.jsonl
```

### 3. `__init__.py` - Package Interface
Makes the data utilities importable as a package.

---

## Quick Start

### Basic Usage (JSONL Input)

```bash
cd "corrected 1-5 experiments/exp1/data"
python preprocess_cli.py input.jsonl output_enriched.jsonl
```

**Before:**
```json
{"question": "Is there a polyp?", "answer": "Yes", "image_id": "123"}
```

**After:**
```json
{
  "question": "Is there a polyp?",
  "answer": "yes",
  "question_type": "yes_no",
  "answer_candidates": ["yes", "no"],
  "image_id": "123"
}
```

---

## Features

### 1. Answer Normalization

```python
from exp1.data import normalize_answer

# Examples
normalize_answer("Yes, there is.")         # → "yes there is"
normalize_answer("  11-20mm  ")           # → "11-20mm"
normalize_answer("Pink/red color")        # → "pinkred color"
normalize_answer("Approximately 5%")      # → "approximately 5%"
```

**What it does:**
- Converts to lowercase
- Strips whitespace
- Removes special characters (except `.`, `-`, `%`)
- Keeps alphanumeric and spaces

---

### 2. Question Type Inference

Automatically categorizes questions into 6 types:

```python
infer_question_type("Is there a polyp?")           # → "yes_no"
infer_question_type("What color is the lesion?")   # → "color"
infer_question_type("What is the size?")           # → "size_numeric"
infer_question_type("How many polyps?")            # → "count_numeric"
infer_question_type("Choose the diagnosis...")     # → "mcq"
infer_question_type("What is the diagnosis?")      # → "open_ended"
```

---

### 3. Answer Candidates

Generates valid answer options for certain question types:

```python
from exp1.data import build_candidates

# Yes/No questions
build_candidates("yes_no", {})
# → ["yes", "no"]

# Color questions
build_candidates("color", {})
# → ["pink", "white", "red", "black", "blue", "brown", 
#    "yellow", "green", "purple", "orange", "gray"]

# MCQ (if options provided)
sample = {"options": ["colonoscopy", "endoscopy", "gastroscopy"]}
build_candidates("mcq", sample)
# → ["colonoscopy", "endoscopy", "gastroscopy"]

# Open-ended (no candidates)
build_candidates("open_ended", {})
# → None
```

---

## CLI Usage

### Help

```bash
python preprocess_cli.py --help
```

### Basic Enrichment

```bash
python preprocess_cli.py train.jsonl train_enriched.jsonl
```

**Output:**
```
Reading from: train.jsonl
Wrote 1000 enriched samples to: train_enriched.jsonl

Question type distribution:
  yes_no          450 ( 45.0%)
  open_ended      350 ( 35.0%)
  count_numeric   150 ( 15.0%)
  size_numeric     30 (  3.0%)
  color            20 (  2.0%)

✓ Successfully processed 1000 samples
```

### Convert JSON to JSONL

```bash
python preprocess_cli.py --from-json data.json data.jsonl
```

### Quiet Mode

```bash
python preprocess_cli.py -q input.jsonl output.jsonl
```

---

## Python API Usage

### As a Package

```python
from exp1.data import normalize_answer, enrich_jsonl, infer_question_type

# Process a single answer
answer = normalize_answer("Yes, there is a polyp.")
print(answer)  # → "yes there is a polyp"

# Enrich a file
enrich_jsonl("train.jsonl", "train_enriched.jsonl")

# Classify question
qtype = infer_question_type("Is there bleeding?")
print(qtype)  # → "yes_no"
```

### Standalone Script

```python
import json
from pathlib import Path

# Add to path if needed
import sys
sys.path.insert(0, 'corrected 1-5 experiments/exp1/data')

from schema import infer_question_type, build_candidates
from preprocess import normalize_answer, enrich_jsonl

# Your code here
```

---

## Real-World Example

### Processing Kvasir-VQA Data

```python
import json
from exp1.data import enrich_jsonl

# Enrich train/val/test splits
enrich_jsonl(
    "datasets/kvasir_raw_6500/train.json",
    "datasets/kvasir_enriched/train.jsonl"
)
enrich_jsonl(
    "datasets/kvasir_raw_6500/val.json",
    "datasets/kvasir_enriched/val.jsonl"
)
enrich_jsonl(
    "datasets/kvasir_raw_6500/test.json",
    "datasets/kvasir_enriched/test.jsonl"
)
```

### Using Enriched Data in Training

```python
import json

# Load enriched data
with open("train_enriched.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        
        question = sample["question"]
        answer = sample["answer"]  # Already normalized
        qtype = sample["question_type"]
        candidates = sample["answer_candidates"]
        
        # Use candidates for constrained generation
        if candidates:
            prediction = model.generate_constrained(
                image=sample["image_path"],
                question=question,
                candidates=candidates
            )
        else:
            prediction = model.generate(
                image=sample["image_path"],
                question=question
            )
```

---

## Input/Output Formats

### Input JSONL Format

Each line is a JSON object:

```jsonl
{"question": "Is there a polyp?", "answer": "yes", "image_id": "123"}
{"question": "What color is it?", "answer": "Pink", "image_id": "124"}
{"question": "How many polyps?", "answer": "2", "image_id": "125"}
```

### Output JSONL Format

Enriched with additional fields:

```jsonl
{"question": "Is there a polyp?", "answer": "yes", "image_id": "123", "question_type": "yes_no", "answer_candidates": ["yes", "no"]}
{"question": "What color is it?", "answer": "pink", "image_id": "124", "question_type": "color", "answer_candidates": ["pink", "white", ...]}
{"question": "How many polyps?", "answer": "2", "image_id": "125", "question_type": "count_numeric", "answer_candidates": null}
```

---

## Advanced: Custom Question Types

### Extend the Schema

Edit `schema.py` to add new types:

```python
def infer_question_type(q: str) -> QuestionType:
    qs = q.lower()
    
    # Add your custom type
    if "instrument" in qs or "tool" in qs:
        return "instrument_type"
    
    # ... existing logic ...
    return "open_ended"

def build_candidates(qtype: QuestionType, sample: Dict):
    # Add candidates for your type
    if qtype == "instrument_type":
        return ["forceps", "scissors", "scalpel", "grasper"]
    
    # ... existing logic ...
```

---

## Troubleshooting

### ImportError: No module named 'exp1'

**Solution 1:** Run from the parent directory
```bash
cd "corrected 1-5 experiments"
python -m exp1.data.preprocess_cli input.jsonl output.jsonl
```

**Solution 2:** Use the standalone CLI
```bash
cd "corrected 1-5 experiments/exp1/data"
python preprocess_cli.py input.jsonl output.jsonl
```

### KeyError: 'question' or 'answer'

Ensure your input JSONL has required fields:
- `question` (str)
- `answer` (str)

Optional fields:
- `image_id` (str)
- `options` (list) - for MCQ questions

---

## Best Practices

1. **Always normalize answers** before training or evaluation
2. **Use enriched data** for type-specific analysis
3. **Leverage candidates** for constrained generation (improves accuracy)
4. **Version your preprocessed data** (track which normalization was used)

---

## See Also

- `schema.py` - Question type definitions
- `analyze_by_type.py` - Analyze results by question type
- `../QUESTION_TYPE_INSIGHTS.md` - Analysis findings
- `README.md` - Package documentation

---

**This preprocessing pipeline ensures consistent, analyzable data for your surgical VQA experiments!** 🔬

