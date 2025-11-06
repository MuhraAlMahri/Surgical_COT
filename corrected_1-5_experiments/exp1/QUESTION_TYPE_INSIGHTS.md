# Question Type Analysis - Key Insights (Exp1)

## 🎯 Overview

Analysis of 8,984 test questions from Exp1 reveals significant **question-type-specific performance patterns** that explain overall model behavior.

---

## 📊 Question Type Distribution

| Type | Count | Percentage | Description |
|------|-------|------------|-------------|
| **Open-Ended** | 3,380 | 37.6% | Free-form medical questions |
| **Yes/No** | 3,071 | 34.2% | Binary decision questions |
| **Count Numeric** | 1,544 | 17.2% | "How many X?" questions |
| **Size Numeric** | 498 | 5.5% | Size/measurement questions |
| **Color** | 491 | 5.5% | Color identification |
| **MCQ** | 0 | 0.0% | Multiple choice (none in test set) |

---

## 🎯 Accuracy by Question Type

### **Performance Ranking:**

| Rank | Type | Accuracy | Performance |
|------|------|----------|-------------|
| **1st** 🏆 | Yes/No | **37.90%** | Nearly 2x overall average |
| **2nd** | Open-Ended | **15.27%** | Below average |
| **3rd** | Count Numeric | **4.15%** | Very poor |
| **4th** | Color | **2.65%** | Nearly failing |
| **5th** ❌ | Size Numeric | **0.00%** | Complete failure |

**Overall Baseline:** 19.56%

---

## 💡 KEY FINDINGS

### **1. Yes/No Questions: The Model's Strength** ✓

```
Type: Yes/No
Questions: 3,071 (34.2%)
Accuracy: 37.90%
Improvement over baseline: +18.34 percentage points
```

**Why it works:**
- Binary decision space (`["yes", "no"]`)
- Clear candidates for evaluation
- Often visually verifiable (e.g., "Is there text?")

**Example Success:**
```
Q: "Is there text?"
Ground Truth: "yes"
Prediction: "Yes, there is text in the image..."
Result: ✓ CORRECT
```

**Opportunity:** 
> Using **constrained generation** with `["yes", "no"]` candidates could push this to 50%+

---

### **2. Size Estimation: Complete Failure** ❌

```
Type: Size Numeric
Questions: 498 (5.5%)
Accuracy: 0.00%
Impact: 498 guaranteed wrong answers
```

**Why it fails:**
- Requires precise measurement understanding
- Ground truth uses specific bins: `"<5mm"`, `"5-10mm"`, `"11-20mm"`
- Model generates vague text: `"approximately 1 cm"`

**Example Failure:**
```
Q: "What is the size of the polyp?"
Ground Truth: "11-20mm"
Prediction: "The size of the polyp is approximately 1 cm."
Result: ✗ WRONG (even though "1 cm = 10mm" is close!)
```

**Root cause:** Format mismatch, not conceptual failure

**Fix:**
1. Post-process predictions to convert units (cm → mm)
2. Map to discrete bins: `["<5mm", "5-10mm", "11-20mm", ">20mm"]`
3. Use constrained generation with these 4 candidates

---

### **3. Counting: High Failure Rate**

```
Type: Count Numeric
Questions: 1,544 (17.2%)
Accuracy: 4.15%
Most common error: Overcounting
```

**Example Failures:**
```
Q: "How many polyps are in the image?"
Ground Truth: "1"
Prediction: "There are three polyps in the image."
Result: ✗ WRONG
```

**Pattern:** Model often predicts 2-3 when answer is 1

**Opportunity:** Constrain to `[0, 1, 2, 3, 4, 5+]` vocabulary

---

### **4. Color: Vocabulary Mismatch**

```
Type: Color
Questions: 491 (5.5%)
Accuracy: 2.65%
Defined vocabulary: 11 colors
```

**Why it fails:**
- Model generates descriptions instead of single colors
- Prediction: "The lesion has a reddish-pink appearance..."
- Expected: "pink"

**Fix:** Use `COLOR_VOCAB` for constrained generation

---

### **5. Open-Ended: Below Average**

```
Type: Open-Ended
Questions: 3,380 (37.6%)
Accuracy: 15.27%
Includes: "What type of X?", "What procedure?", etc.
```

**Challenges:**
- Most diverse answer space
- Requires medical terminology
- Example: "What type of polyp?" → "paris is" (Paris classification)

---

## 🚀 IMPROVEMENT OPPORTUNITIES

### **Strategy 1: Constrained Generation by Type**

Implement type-specific output constraints:

```python
from schema import infer_question_type, build_candidates

def predict(question, image):
    qtype = infer_question_type(question)
    candidates = build_candidates(qtype, sample)
    
    if candidates:
        # Force model to choose from valid set
        return model.generate_constrained(image, question, candidates)
    else:
        # Free-form generation
        return model.generate(image, question)
```

**Expected gains:**
- Yes/No: 37.90% → **50%** (+12%)
- Color: 2.65% → **15%** (+12%)
- Size: 0.00% → **20%** (+20%) with bin mapping

---

### **Strategy 2: Post-Processing Pipeline**

```python
def post_process(prediction, qtype):
    if qtype == "size_numeric":
        # Convert "1 cm" → "10mm" → map to "5-10mm"
        return map_to_size_bin(prediction)
    
    if qtype == "count_numeric":
        # Extract number: "three polyps" → "3"
        return extract_number(prediction)
    
    if qtype == "color":
        # Extract first color word
        return extract_color(prediction, COLOR_VOCAB)
    
    return prediction
```

**Expected gains:**
- Size: 0.00% → **15%** (+15%)
- Count: 4.15% → **10%** (+6%)
- Color: 2.65% → **8%** (+5%)

---

### **Strategy 3: Type-Weighted Evaluation**

Current overall accuracy (19.56%) is misleading because:
- 37.90% on yes/no questions (1/3 of dataset)
- 0.00% on size questions (5% of dataset)

**Weighted accuracy by importance:**
```
Clinical Impact Weight:
- Size/Count: 3x (critical for diagnosis)
- Yes/No: 1x (basic screening)
- Color: 2x (pathology indicator)
```

This reveals that model performs poorly on **clinically important** questions.

---

## 📈 PROJECTED IMPACT

### **If All Strategies Applied:**

| Type | Current | With Fixes | Improvement |
|------|---------|------------|-------------|
| Yes/No | 37.90% | **50%** | +12% |
| Open-Ended | 15.27% | **18%** | +3% |
| Count | 4.15% | **10%** | +6% |
| Size | 0.00% | **20%** | +20% |
| Color | 2.65% | **15%** | +12% |

**Overall projected:** 19.56% → **28-30%** (+8-10 percentage points)

---

## 🎓 FOR YOUR PRESENTATION

### **Key Message:**

> "Analysis by question type reveals that the model's 19.56% overall accuracy masks **significant variation**: 37.90% on yes/no questions but 0% on size estimation. This isn't random—it reflects the **structural challenge** of mapping verbose language model outputs to structured medical vocabularies."

### **Slide Content:**

```
Question Type Analysis

Performance by Type:
  ✓ Yes/No:        37.90%  (2x baseline)
  ✗ Size:           0.00%  (complete failure)
  ✗ Counting:       4.15%  (critical gap)

Key Insight:
  Problem ≠ Visual understanding
  Problem = Output format mismatch

Solution:
  • Constrained generation
  • Post-processing pipeline
  • Type-specific evaluation
```

---

## 📁 FILES CREATED

```
corrected 1-5 experiments/exp1/data/
├── schema.py                    # Question type definitions
├── analyze_by_type.py           # Analysis script
├── question_type_stats.json     # Detailed statistics
└── README.md                    # Documentation
```

**Usage:**
```bash
cd "corrected 1-5 experiments/exp1/data"
python3 analyze_by_type.py
```

---

## 🔬 SCIENTIFIC CONTRIBUTION

This analysis reveals that:

1. **Overall accuracy is misleading** - hides type-specific patterns
2. **Constrained decoding is critical** - for structured medical QA
3. **Format standardization matters** - more than model capacity
4. **Question type should be a standard metric** - for VQA evaluation

**Recommendation:** Future surgical VQA papers should report **type-stratified accuracy**, not just overall metrics.

---

**This granular analysis transforms a "disappointing 20%" into actionable insights for improvement!** 🚀

