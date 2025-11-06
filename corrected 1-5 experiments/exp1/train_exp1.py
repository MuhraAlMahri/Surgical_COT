import argparse
import yaml
import os
import sys
from pathlib import Path

# Add exp1 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import VQASFTDataset, collate
import torch


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # preprocess jsonl to add question_type, candidates, normalized answers
    # Import from data submodule
    sys.path.insert(0, str(Path(__file__).parent / "data"))
    from preprocess import enrich_jsonl
    
    for split in ["train_jsonl", "val_jsonl"]:
        inp = cfg["data"][split]
        outp = inp.replace(".jsonl", ".enriched.jsonl")
        if not os.path.exists(outp):
            print(f"Enriching {inp} -> {outp}")
            enrich_jsonl(inp, outp)
        cfg["data"][split] = outp
    
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    
    if cfg.get("vision_frozen", True):
        for n, p in model.named_parameters():
            if "vision_tower" in n or "visual" in n:
                p.requires_grad = False
    
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    
    train_ds = VQASFTDataset(
        cfg["data"]["train_jsonl"],
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"]
    )
    val_ds = VQASFTDataset(
        cfg["data"]["val_jsonl"],
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"]
    )
    
    args = TrainingArguments(
        output_dir="corrected 1-5 experiments/exp1/outputs",
        per_device_train_batch_size=cfg["train"]["train_bs"],
        per_device_eval_batch_size=cfg["train"]["eval_bs"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        num_train_epochs=cfg["train"]["epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        bf16=cfg["train"]["bf16"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["train"]["save_steps"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        # exact-match on normalized strings; numeric tolerance from cfg
        # We won't decode here; evaluation will be done by a separate generate+eval script.
        return {}
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    
    trainer.train()


if __name__ == "__main__":
    import sys
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "corrected 1-5 experiments/exp1/config_exp1.yaml"
    main(cfg_file)

