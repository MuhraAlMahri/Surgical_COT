import torch
from transformers.generation.logits_process import LogitsProcessor


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids):
        super().__init__()
        self.allowed = set(allowed_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(list(self.allowed), device=scores.device)
        mask[..., idx] = scores[..., idx]
        return mask


def build_allowed_ids(tokenizer, candidates):
    ids = set()
    for a in candidates:
        toks = tokenizer(a, add_special_tokens=False).input_ids
        # allow multi-token words by allowing the first token; generation will be greedy for short outputs.
        for t in toks:
            ids.add(t)
    # Always allow EOS
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return list(ids)

