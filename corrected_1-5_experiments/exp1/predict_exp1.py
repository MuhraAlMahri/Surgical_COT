import json
import torch
import yaml
import sys
from pathlib import Path

# Add exp1 to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig
from peft import PeftModel
from PIL import Image
from templates import prompt_block
from constraints import AllowedTokensLogitsProcessor, build_allowed_ids


def load_model(cfg, script_dir):
    model_name = cfg["model_name"]
    output_dir = script_dir / "outputs"
    
    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval().cuda()
    
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = proc.tokenizer
    
    return model, tok, proc


def generate_answer(model, tok, proc, img_path, question, qtype, candidates=None, max_new=4):
    prompt = prompt_block(qtype, question, candidates)
    enc = proc(
        text=prompt,
        images=Image.open(img_path).convert("RGB"),
        return_tensors="pt"
    ).to(model.device)
    
    logits_processors = []
    if candidates:
        allowed = build_allowed_ids(tok, candidates)
        logits_processors = [AllowedTokensLogitsProcessor(allowed)]
    
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        eos_token_id=tok.eos_token_id,
        logits_processor=logits_processors
    )
    
    text = tok.decode(out[0], skip_special_tokens=True)
    # take last line after "Answer:" or just the tail
    ans = text.split("Answer:")[-1].strip().split("\n")[0].strip().lower()
    return ans


def main():
    script_dir = Path(__file__).parent
    cfg_path = script_dir / "config_exp1.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    model, tok, proc = load_model(cfg, script_dir)
    
    # Resolve val_path relative to base directory
    base_dir = script_dir.parent
    val_path = base_dir / cfg["data"]["val_jsonl"]
    img_root = cfg["data"]["image_root"]
    
    preds = []
    with open(str(val_path)) as f:
        for line in f:
            ex = json.loads(line)
            img_file = ex.get('image') or ex.get('image_filename')
            ans = generate_answer(
                model, tok, proc,
                f"{img_root}/{img_file}",
                ex["question"],
                ex["question_type"],
                ex.get("answer_candidates")
            )
            preds.append({
                "id": ex.get("id") or ex.get("image_id"),
                "pred": ans,
                "gt": ex["answer"],
                "qtype": ex["question_type"]
            })
    
    output_file = script_dir / "outputs/predictions.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    
    print(f"✓ Generated {len(preds)} predictions -> {output_file}")


if __name__ == "__main__":
    main()

