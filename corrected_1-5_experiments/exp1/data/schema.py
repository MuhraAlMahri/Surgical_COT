from typing import List, Optional, Literal, Dict

QuestionType = Literal["yes_no", "color", "size_numeric", "count_numeric", "mcq", "open_ended"]

COLOR_VOCAB = ["pink", "white", "red", "black", "blue", "brown", "yellow", "green", "purple", "orange", "gray"]

def infer_question_type(q: str) -> QuestionType:
    qs = q.lower()
    if "yes or no" in qs or qs.startswith("is ") or qs.startswith("are "):
        return "yes_no"
    if "color" in qs or "colour" in qs:
        return "color"
    if "size" in qs or "length" in qs or "diameter" in qs:
        return "size_numeric"
    if "how many" in qs or "count" in qs:
        return "count_numeric"
    if "choose" in qs or "options" in qs or "mcq" in qs:
        return "mcq"
    return "open_ended"

def build_candidates(qtype: QuestionType, sample: Dict) -> Optional[List[str]]:
    if qtype == "yes_no":
        return ["yes", "no"]
    if qtype == "color":
        return COLOR_VOCAB
    if qtype == "mcq":
        return sample.get("options")  # expects list like ["A", "B", "C"] or real strings
    return None

