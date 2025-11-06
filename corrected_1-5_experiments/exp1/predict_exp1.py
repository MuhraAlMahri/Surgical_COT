import os
import sys
import json
import yaml
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig
from peft import PeftModel
from PIL import Image
from templates import build_conversation_inference
from constraints import AllowedTokensLogitsProcessor, build_allowed_ids


def load_model(cfg, script_dir):
    model_name = cfg["model_name"]
    output_dir = script_dir / "outputs"
    
    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    
    # Load LoRA weights if they exist
    if (output_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(output_dir))
        print(f"Loaded LoRA adapter from {output_dir}")
    else:
        print(f"Warning: No LoRA adapter found at {output_dir}, using base model")
    
    model.eval().cuda()
    
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = proc.tokenizer
    
    return model, tok, proc


def generate_answer(model, tok, proc, img_path, question, qtype, candidates=None, max_new=4):
    """Generate answer with optional constrained decoding."""
    # Build conversation for inference (no answer, no sentinels)
    conversation = build_conversation_inference(qtype, question, candidates)
    
    # Apply chat template
    text = proc.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True  # Add generation prompt for inference
    )
    
    # Process with image
    enc = proc(
        text=[text],
        images=[Image.open(img_path).convert("RGB")],
        return_tensors="pt"
    ).to(model.device)
    
    # Setup constrained decoding if candidates provided
    logits_processor = None
    if candidates and qtype in ("yes_no", "color", "mcq"):
        allowed_ids = build_allowed_ids(tok, candidates)
        logits_processor = [AllowedTokensLogitsProcessor(allowed_ids)]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            logits_processor=logits_processor
        )
    
    # Decode only the generated part
    prompt_len = enc["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    answer = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    return answer


def main():
    script_dir = Path(__file__).parent
    cfg_path = script_dir / "config_exp1.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    
    print("Loading model...")
    model, tok, proc = load_model(cfg, script_dir)
    
    base_dir = script_dir.parent
    val_path = base_dir / cfg["data"]["val_jsonl"]
    # Use enriched data
    val_enriched = str(val_path).replace(".jsonl", ".enriched.jsonl")
    if not Path(val_enriched).exists():
        print(f"Warning: Enriched validation data not found at {val_enriched}")
        print("Using original validation data instead")
        val_enriched = str(val_path)
    
    img_root = cfg["data"]["image_root"]
    
    print(f"Generating predictions on {val_enriched}...")
    preds = []
    
    with open(val_enriched) as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print(f"  Processed {idx} samples...")
            
            ex = json.loads(line)
            img_file = ex.get('image') or ex.get('image_filename') or ex.get('image_id')
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_file = f"{img_file}.jpg"
            img_path = f"{img_root}/{img_file}"
            
            ans = generate_answer(
                model, tok, proc,
                img_path,
                ex["question"],
                ex["question_type"],
                ex.get("answer_candidates")
            )
            
            preds.append({
                "id": ex.get("id"),
                "pred": ans,
                "gt": ex["answer"],
                "qtype": ex["question_type"]
            })
    
    output_file = script_dir / "outputs/predictions.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    
    print(f"Done! Generated {len(preds)} predictions")


if __name__ == "__main__":
    import torch
    main()
