# Data for Presentation Charts & Visuals

## 📊 CHART 1: Overall Accuracy Comparison

**Type:** Horizontal Bar Chart or Column Chart

```
Experiment                  | Accuracy | Color Suggestion
----------------------------|----------|------------------
Exp1: Random Baseline       | 20.31%   | Gray
Exp2: Qwen Reordered        | 20.27%   | Light Blue
Exp3: CXR-TREK Sequential   | 21.14%   | Blue
Exp4: Curriculum Learning   | 21.32%   | Green (BEST)
Exp5: Sequential CoT        | 21.08%   | Orange
```

**PowerPoint/Google Slides Data Entry:**
- Copy-paste into chart data editor
- Set Y-axis range: 19% to 22%
- Add data labels on bars
- Highlight Exp4 with different color/pattern

---

## 📊 CHART 2: Stage-wise Performance Breakdown

**Type:** Grouped Bar Chart

```
Stage         | Exp3     | Exp4     | Exp5
--------------|----------|----------|----------
Stage 1       | 33.04%   | 33.13%   | 33.62%
Stage 2       | 14.33%   | 14.55%   | 13.90%
Stage 3       | 0.00%    | 0.00%    | 0.00%
```

**Visualization Tips:**
- Group by stage (X-axis)
- Three bars per group (one per experiment)
- Use consistent colors: Exp3=Blue, Exp4=Green, Exp5=Orange
- Add horizontal line at 20% to show "baseline"
- Annotate Stage 3 with "Insufficient data (n=6)"

---

## 📊 CHART 3: Dataset Stage Distribution

**Type:** Pie Chart or Donut Chart

```
Stage                        | Samples | Percentage
-----------------------------|---------|------------
Stage 1 (Quality Control)    | 3,275   | 36.47%
Stage 2 (Findings)           | 5,703   | 63.47%
Stage 3 (Clinical Reasoning) | 6       | 0.07%
----------------------------- ---------|------------
TOTAL                        | 8,984   | 100.00%
```

**Color Suggestions:**
- Stage 1: Light Green
- Stage 2: Medium Blue
- Stage 3: Red (to highlight scarcity)

---

## 📊 CHART 4: Training Architecture Comparison

**Type:** Comparison Table

```
Feature              | Exp1    | Exp2    | Exp3        | Exp4       | Exp5
---------------------|---------|---------|-------------|------------|-------------
# of Models          | 1       | 1       | 3           | 1          | 1
Training Strategy    | Random  | Random  | Stage-spec. | Curriculum | Standard
Data Ordering        | Random  | Qwen    | By Stage    | Easy→Hard  | Mixed
Inference Type       | Direct  | Direct  | Stage-spec. | Direct     | Cascaded
Training Time (hrs)  | ~3      | ~3      | ~12         | ~4         | ~3
Best Stage 1 Acc     | -       | -       | 33.04%      | 33.13%     | 33.62%
Best Stage 2 Acc     | -       | -       | 14.33%      | 14.55%     | 13.90%
Overall Accuracy     | 20.31%  | 20.27%  | 21.14%      | 21.32%     | 21.08%
```

---

## 📊 CHART 5: Performance vs Baseline Improvement

**Type:** Line Chart or Bar Chart with Baseline

```
Experiment    | Absolute Accuracy | Improvement over Exp1
--------------|-------------------|-----------------------
Exp1 (Base)   | 20.31%            | 0.00% (baseline)
Exp2          | 20.27%            | -0.04%
Exp3          | 21.14%            | +0.83%
Exp4          | 21.32%            | +1.01%
Exp5          | 21.08%            | +0.77%
```

---

## 📊 CHART 6: Stage Complexity vs Performance

**Type:** Scatter Plot or Line Chart

```
Stage                    | Avg Accuracy | Complexity Score (1-10)
-------------------------|--------------|------------------------
Stage 1 (Quality)        | 33.27%       | 3
Stage 2 (Findings)       | 14.26%       | 7
Stage 3 (Clinical)       | 0.00%        | 10
```

**Insight to highlight:**
> "Clear inverse correlation: as clinical complexity increases, model performance drops dramatically"

---

## 📋 TABLE 1: Experiment Summary Matrix

**Type:** Comparison Matrix for Slide**

```
┌────────┬──────────────────┬─────────────────────┬─────────┬──────────────────────────┐
│ Exp    │ Method           │ Key Innovation      │ Accuracy│ Pros / Cons              │
├────────┼──────────────────┼─────────────────────┼─────────┼──────────────────────────┤
│ Exp1   │ Random           │ Baseline            │ 20.31%  │ ✓ Simple                 │
│        │ Baseline         │                     │         │ ✗ No structure           │
├────────┼──────────────────┼─────────────────────┼─────────┼──────────────────────────┤
│ Exp2   │ Qwen             │ LLM-based ordering  │ 20.27%  │ ✓ Minimal effort         │
│        │ Reordered        │                     │         │ ✗ No improvement         │
├────────┼──────────────────┼─────────────────────┼─────────┼──────────────────────────┤
│ Exp3   │ CXR-TREK         │ 3 specialized models│ 21.14%  │ ✓ Stage-specific         │
│        │ Sequential       │                     │         │ ✗ 3x storage cost        │
├────────┼──────────────────┼─────────────────────┼─────────┼──────────────────────────┤
│ Exp4   │ Curriculum       │ Progressive training│ 21.32%  │ ✓ Best accuracy          │
│        │ Learning         │                     │         │ ✓ Single model           │
│        │                  │                     │         │ ✓ Efficient              │
├────────┼──────────────────┼─────────────────────┼─────────┼──────────────────────────┤
│ Exp5   │ Sequential       │ Cascaded reasoning  │ 21.08%  │ ✓ Mimics clinicians      │
│        │ CoT              │                     │         │ ✗ Error propagation      │
└────────┴──────────────────┴─────────────────────┴─────────┴──────────────────────────┘
```

---

## 📊 CHART 7: Question Type Performance Analysis

**Type:** Horizontal Bar Chart

```
Question Type                 | Accuracy | Example Question
------------------------------|----------|----------------------------------
Yes/No Questions              | ~45%     | "Is there a polyp?"
Presence Detection            | ~35%     | "Is there bleeding?"
Artifact Detection            | ~32%     | "Is there text?"
Anatomical Identification     | ~18%     | "What organ is visible?"
Size Estimation               | ~8%      | "What is the size of the polyp?"
Medical Classification        | ~5%      | "What type of polyp?"
```

**Note:** These are approximate values based on manual analysis of results

---

## 🎯 KEY STATISTICS TO HIGHLIGHT

### Dataset Statistics:
- **Total Test Samples:** 8,984
- **Total Images:** 6,500
- **Split Ratio:** 70% train / 15% val / 15% test
- **Question Types:** Quality (36.5%), Findings (63.5%), Clinical (0.07%)

### Model Statistics:
- **Base Model:** Qwen2-VL-7B-Instruct
- **Parameters:** 7 billion
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 64
- **Trainable Parameters:** ~20M (0.3% of total)

### Training Statistics:
- **Total GPU Hours:** ~50 hours
- **Average Training Time per Experiment:** 3-12 hours
- **Batch Size:** 4
- **Learning Rate:** 2e-4
- **Epochs:** 3

### Performance Statistics:
- **Best Overall Accuracy:** 21.32% (Exp4)
- **Improvement over Baseline:** +1.01%
- **Best Stage 1 Accuracy:** 33.62% (Exp5)
- **Best Stage 2 Accuracy:** 14.55% (Exp4)
- **Average Inference Time:** ~2 seconds per question

---

## 📊 CHART 8: Training Efficiency Comparison

**Type:** Scatter Plot (X=Training Time, Y=Accuracy)**

```
Experiment | Training Time (hrs) | Accuracy | Efficiency Score
-----------|---------------------|----------|------------------
Exp1       | 3                   | 20.31%   | 6.77
Exp2       | 3                   | 20.27%   | 6.76
Exp3       | 12                  | 21.14%   | 1.76
Exp4       | 4                   | 21.32%   | 5.33 ⭐
Exp5       | 3                   | 21.08%   | 7.03 ⭐⭐
```

*Efficiency Score = Accuracy / Training Time*

**Insight:**
> "Exp5 achieves competitive accuracy with minimal training time, while Exp4 offers the best accuracy with reasonable efficiency."

---

## 🎨 COLOR PALETTE RECOMMENDATIONS

For consistency across all charts:

```
Primary Colors:
- Exp1 (Baseline):    #95A5A6 (Gray)
- Exp2 (Reordered):   #3498DB (Light Blue)
- Exp3 (Sequential):  #2C3E50 (Dark Blue)
- Exp4 (Curriculum):  #27AE60 (Green) ⭐ BEST
- Exp5 (CoT):         #E67E22 (Orange)

Stage Colors:
- Stage 1:            #2ECC71 (Light Green)
- Stage 2:            #3498DB (Medium Blue)
- Stage 3:            #E74C3C (Red - highlights problem)

Status Colors:
- Success (✓):        #27AE60 (Green)
- Failure (✗):        #E74C3C (Red)
- Warning (⚠):        #F39C12 (Orange)
```

---

## 💡 VISUAL DESIGN TIPS

1. **Use Icons:**
   - 🏆 for best performer
   - ⚠️ for challenges/limitations
   - ✓ for successes
   - ✗ for failures

2. **Highlight Key Numbers:**
   - Make the best accuracy bold and larger font
   - Use color-coded boxes for emphasis

3. **Consistent Layout:**
   - Same color scheme across all slides
   - Same font sizes for similar elements
   - Align charts and tables consistently

4. **White Space:**
   - Don't overcrowd slides
   - One main point per slide
   - Use progressive reveal for complex charts

---

## 📸 SUGGESTED SLIDE LAYOUTS

### Results Slide Layout:
```
┌─────────────────────────────────────────────────┐
│ Slide Title: "Overall Performance Comparison"  │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Bar Chart - 60% of slide]                    │
│                                                 │
│                                                 │
├─────────────────────────────────────────────────┤
│ Key Insight (text box - 40%):                  │
│ "Curriculum learning achieved 21.32% accuracy, │
│  outperforming baselines by +1.01%"            │
└─────────────────────────────────────────────────┘
```

### Example Slide Layout:
```
┌─────────────────────────────────────────────────┐
│ Slide Title: "Success Example - Stage 1"      │
├──────────────────┬──────────────────────────────┤
│                  │  Question: "Is there text?" │
│  [Image 50%]     │  Ground Truth: "yes"        │
│                  │  Prediction: "Yes, there... │
│                  │  Extracted: "yes" ✓         │
└──────────────────┴──────────────────────────────┘
```

---

## ⚡ QUICK COPY-PASTE DATA

For easy chart creation, here's CSV format:

**Overall Accuracy (CSV):**
```
Experiment,Accuracy
Exp1 Random,20.31
Exp2 Reordered,20.27
Exp3 Sequential,21.14
Exp4 Curriculum,21.32
Exp5 CoT,21.08
```

**Stage-wise Performance (CSV):**
```
Stage,Exp3,Exp4,Exp5
Stage 1,33.04,33.13,33.62
Stage 2,14.33,14.55,13.90
Stage 3,0.00,0.00,0.00
```

**Dataset Distribution (CSV):**
```
Stage,Samples,Percentage
Stage 1,3275,36.47
Stage 2,5703,63.47
Stage 3,6,0.07
```

---

**Use these charts to create professional, data-driven visuals for your presentation!** 📊

