import json
import re
from pathlib import Path
from .schema import infer_question_type, build_candidates


def normalize_answer(ans: str) -> str:
    x = ans.strip().lower()
    x = re.sub(r"[^\w\.\-\% ]+", "", x)  # keep simple tokens
    return x


def enrich_jsonl(in_path, out_path):
    out = []
    with open(in_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            gt = normalize_answer(ex["answer"])
            qtype = ex.get("question_type") or infer_question_type(q)
            ex["question_type"] = qtype
            ex["answer"] = gt
            ex["answer_candidates"] = build_candidates(qtype, ex)
            out.append(ex)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")

