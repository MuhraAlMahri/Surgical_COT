import json
import math
import re
import yaml
from collections import defaultdict


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[^\w\.\- ]+", "", s)
    return s


def maybe_float(s):
    try:
        return float(s)
    except:
        return None


def evaluate(pred_path, cfg_path="corrected_1-5_experiments/exp1/config_exp1.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    eps_rel = cfg["eval"]["numeric_rel_tol"]
    eps_abs = cfg["eval"]["numeric_abs_tol"]
    
    counts = defaultdict(int)
    correct = defaultdict(int)
    
    for line in open(pred_path):
        ex = json.loads(line)
        q = ex["qtype"]
        gt = norm(ex["gt"])
        pr = norm(ex["pred"])
        counts[q] += 1
        ok = False
        
        if q in ("yes_no", "color", "mcq", "open_ended"):
            ok = (gt == pr)
        elif q in ("size_numeric", "count_numeric"):
            gtf = maybe_float(re.findall(r"[-+]?\d*\.?\d+", gt)[0]) if re.findall(r"[-+]?\d*\.?\d+", gt) else None
            prf = maybe_float(re.findall(r"[-+]?\d*\.?\d+", pr)[0]) if re.findall(r"[-+]?\d*\.?\d+", pr) else None
            if gtf is not None and prf is not None:
                ok = abs(gtf - prf) <= max(eps_abs, eps_rel * max(1.0, abs(gtf)))
        
        correct[q] += int(ok)
    
    per_type = {k: (correct[k] / counts[k] if counts[k] else 0.0) for k in counts}
    micro = (sum(correct.values()) / sum(counts.values())) if counts else 0.0
    
    return per_type, micro, counts, correct


if __name__ == "__main__":
    per_type, micro, counts, correct = evaluate("corrected_1-5_experiments/exp1/outputs/predictions.jsonl")
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - EXP1")
    print("=" * 80)
    print(f"\n{'Question Type':<20} {'Count':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    for qtype in sorted(per_type.keys()):
        acc = per_type[qtype] * 100
        print(f"{qtype:<20} {counts[qtype]:<10} {correct[qtype]:<10} {acc:<10.2f}%")
    
    print("-" * 80)
    print(f"{'OVERALL (micro)':<20} {sum(counts.values()):<10} {sum(correct.values()):<10} {micro * 100:<10.2f}%")
    print("=" * 80)
    
    print("\nPer-type accuracy dict:", per_type)
    print(f"Overall (micro): {micro:.4f}")

