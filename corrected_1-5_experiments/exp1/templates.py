def prompt_block(question_type, question, answer_candidates=None, answer=None, for_training=True):
    """
    Build instruction prompt for surgical VQA.
    
    Args:
        question_type: Type of question (yes_no, color, etc.)
        question: The question text
        answer_candidates: Optional list of valid answers
        answer: Ground truth answer (only for training)
        for_training: If True, include answer with sentinels. If False, omit for inference.
    
    Returns:
        Formatted prompt string
    """
    lines = [
        "<image>",
        "System: You are a surgical VQA assistant. Answer with a single word/number when possible.",
        "User:",
        f"Question type: {question_type}",
        f"Question: {question}"
    ]
    
    if answer_candidates:
        cand = ", ".join(answer_candidates)
        lines.append(f"Valid answers: {cand}")
    
    lines.append("Assistant: Answer:")
    
    if for_training and answer is not None:
        # Add answer with sentinels for training
        lines.append(f"<ANS>{answer}</ANS>")
    
    return "\n".join(lines)


def prompt_block_inference(question_type, question, answer_candidates=None):
    """Convenience function for inference (no answer, no sentinels)."""
    return prompt_block(question_type, question, answer_candidates, answer=None, for_training=False)
