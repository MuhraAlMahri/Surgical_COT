def prompt_block(question_type, question, answer_candidates=None):
    lines = [
        "System: You are a surgical VQA assistant. Answer concisely with only the final answer.",
        "User:",
        f"Question type: {question_type}",
        f"Question: {question}"
    ]
    if answer_candidates:
        cand = ", ".join(answer_candidates)
        lines.append(f"Valid answers: {cand}")
    lines.append("Assistant: Answer:")
    return "\n".join(lines)

