#!/usr/bin/env python3
"""
Sanity overfit script - sample 64 train examples, 200 steps.
Expect loss < ~2 if masking is correct and model can memorize short answers.
"""

import yaml
import sys
import json
from pathlib import Path

# Add exp1 to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import VQASFTDataset, collate
import torch


def create_tiny_dataset(jsonl_path, output_path, n_samples=64):
    """Create a tiny subset for overfitting test."""
    samples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    
    return len(samples)


def main():
    cfg_path = "corrected_1-5_experiments/exp1/config_exp1.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # Create tiny datasets
    data_dir = Path(__file__).parent / "data"
    sys.path.insert(0, str(data_dir))
    
    # Import schema module
    import schema
    import json
    import re
    from pathlib import Path as PathlibPath
    
    def normalize_answer_local(ans):
        x = ans.strip().lower()
        x = re.sub(r"[^\w\.\-\% ]+", "", x)
        return x
    
    def enrich_jsonl_local(in_path, out_path):
        out = []
        with open(in_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                q = ex["question"]
                gt = normalize_answer_local(ex["answer"])
                qtype = ex.get("question_type") or schema.infer_question_type(q)
                ex["question_type"] = qtype
                ex["answer"] = gt
                ex["answer_candidates"] = schema.build_candidates(qtype, ex)
                out.append(ex)
        PathlibPath(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in out:
                f.write(json.dumps(ex) + "\n")
    
    enrich_jsonl = enrich_jsonl_local
    
    train_input = cfg["data"]["train_jsonl"]
    train_enriched = train_input.replace(".jsonl", ".enriched.jsonl")
    
    if not Path(train_enriched).exists():
        print(f"Enriching {train_input}...")
        enrich_jsonl(train_input, train_enriched)
    
    # Create tiny subset
    tiny_train = "corrected_1-5_experiments/exp1/outputs/tiny_train.jsonl"
    n_train = create_tiny_dataset(train_enriched, tiny_train, n_samples=64)
    
    tiny_val = "corrected_1-5_experiments/exp1/outputs/tiny_val.jsonl"
    n_val = create_tiny_dataset(train_enriched, tiny_val, n_samples=16)
    
    print(f"\n{'='*80}")
    print(f"SANITY OVERFIT TEST")
    print(f"{'='*80}")
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    print(f"Expected: Loss should drop below 2.0 within 200 steps")
    print(f"{'='*80}\n")
    
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    
    # Freeze vision tower
    for n, p in model.named_parameters():
        if "vision_tower" in n or "visual" in n:
            p.requires_grad = False
    
    # Add LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=0.0,  # No dropout for overfitting
        target_modules=cfg["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    
    # Create datasets
    train_ds = VQASFTDataset(tiny_train, cfg["data"]["image_root"], model_name, cfg["train"]["max_seq_len"])
    val_ds = VQASFTDataset(tiny_val, cfg["data"]["image_root"], model_name, cfg["train"]["max_seq_len"])
    
    # Training args optimized for overfitting
    args = TrainingArguments(
        output_dir="corrected_1-5_experiments/exp1/outputs/sanity_overfit",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,  # Higher LR for faster overfitting
        weight_decay=0.0,  # No regularization
        max_steps=200,
        warmup_steps=10,
        bf16=cfg["train"]["bf16"],
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        gradient_checkpointing=False,
        report_to="none",
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    
    print("Starting sanity overfit training...")
    print("Watch for:")
    print("  - Training loss should drop rapidly")
    print("  - Final loss < 2.0 indicates proper label masking")
    print("  - Loss stuck at ~7.2 indicates masking bug")
    print()
    
    trainer.train()
    
    # Check final loss
    import json
    state_file = Path(args.output_dir) / "trainer_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            if state.get('log_history'):
                final_loss = None
                for entry in reversed(state['log_history']):
                    if 'loss' in entry:
                        final_loss = entry['loss']
                        break
                
                print(f"\n{'='*80}")
                print(f"SANITY CHECK RESULT")
                print(f"{'='*80}")
                if final_loss is not None:
                    print(f"Final training loss: {final_loss:.4f}")
                    if final_loss < 2.0:
                        print("✓ PASS - Loss dropped as expected. Label masking is working!")
                    elif final_loss > 5.0:
                        print("✗ FAIL - Loss too high. Check label masking implementation!")
                    else:
                        print("⚠ WARNING - Loss intermediate. May need more steps or check implementation.")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

