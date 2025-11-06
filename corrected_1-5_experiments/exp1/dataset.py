import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor
from templates import prompt_block


class VQASFTDataset(Dataset):
    def __init__(self, jsonl_path, image_root, model_name, max_len=512):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.image_root = Path(image_root)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ex = self.samples[i]
        # Handle different image field names
        img_file = ex.get('image') or ex.get('image_filename') or ex.get('image_id')
        if not img_file:
            raise KeyError(f"No image field found in sample. Available fields: {ex.keys()}")
        # Add .jpg extension if needed
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_file = f"{img_file}.jpg"
        img_path = self.image_root / img_file
        img = Image.open(str(img_path).replace("//", "/")).convert("RGB")
        prompt = prompt_block(ex["question_type"], ex["question"], ex.get("answer_candidates"))
        enc = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        )
        input_ids = enc["input_ids"][0]
        pixel_values = enc["pixel_values"][0]
        
        # labels: mask ALL prompt tokens with -100, then append answer tokens with labels
        tok = self.processor.tokenizer
        ans_ids = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ans_ids = torch.tensor(ans_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, fill_value=-100)
        input_ids = torch.cat([input_ids, ans_ids], dim=0)
        attn = torch.cat([enc["attention_mask"][0], torch.ones_like(ans_ids)], dim=0)
        labels = torch.cat([labels, ans_ids], dim=0)
        
        # clip if we exceeded max_len
        input_ids = input_ids[:self.max_len]
        attn = attn[:self.max_len]
        labels = labels[:self.max_len]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": pixel_values,
            "labels": labels
        }


def collate(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    return out

