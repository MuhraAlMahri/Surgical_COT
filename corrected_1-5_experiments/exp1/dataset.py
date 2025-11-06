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
        
        # Ensure right padding for training
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ex = self.samples[i]
        
        # Load image
        img_file = ex.get('image') or ex.get('image_filename') or ex.get('image_id')
        if not img_file:
            raise KeyError(f"No image field found in sample. Available fields: {ex.keys()}")
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_file = f"{img_file}.jpg"
        img_path = self.image_root / img_file
        img = Image.open(str(img_path).replace("//", "/")).convert("RGB")
        
        # Build prompt WITH answer (wrapped in sentinels)
        prompt_with_answer = prompt_block(
            ex["question_type"],
            ex["question"],
            ex.get("answer_candidates"),
            answer=ex["answer"],  # Include ground truth
            for_training=True
        )
        
        # Process ONCE with image and full text (including answer)
        enc = self.processor(
            text=[prompt_with_answer],
            images=[img],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        )
        
        # Extract tensors
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values = enc.get("pixel_values", [None])[0]
        image_grid_thw = enc.get("image_grid_thw")
        
        # Sentinel-based label masking
        # Find <ANS> and </ANS> token positions
        tok = self.processor.tokenizer
        
        # Tokenize sentinels to find their IDs
        ans_start_tokens = tok("<ANS>", add_special_tokens=False)["input_ids"]
        ans_end_tokens = tok("</ANS>", add_special_tokens=False)["input_ids"]
        
        # Initialize labels: all -100 (masked)
        labels = torch.full_like(input_ids, fill_value=-100)
        
        # Find sentinel positions in input_ids
        input_ids_list = input_ids.tolist()
        
        # Search for <ANS> start
        ans_start_idx = None
        ans_end_idx = None
        
        for idx in range(len(input_ids_list) - len(ans_start_tokens) + 1):
            if input_ids_list[idx:idx+len(ans_start_tokens)] == ans_start_tokens:
                ans_start_idx = idx + len(ans_start_tokens)  # Start AFTER <ANS>
                break
        
        # Search for </ANS> end
        if ans_start_idx:
            for idx in range(ans_start_idx, len(input_ids_list) - len(ans_end_tokens) + 1):
                if input_ids_list[idx:idx+len(ans_end_tokens)] == ans_end_tokens:
                    ans_end_idx = idx  # End BEFORE </ANS>
                    break
        
        # Apply masking: only supervise answer tokens (between sentinels)
        if ans_start_idx and ans_end_idx and ans_start_idx < ans_end_idx:
            # Unmask answer span
            labels[ans_start_idx:ans_end_idx] = input_ids[ans_start_idx:ans_end_idx]
        else:
            # Debugging: print what went wrong
            print(f"WARNING: Could not find answer sentinels in sample {i}")
            print(f"  Decoded text: {tok.decode(input_ids, skip_special_tokens=False)[:200]}...")
            print(f"  ans_start_idx: {ans_start_idx}, ans_end_idx: {ans_end_idx}")
        
        # Build result dictionary
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add vision-related tensors if available
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            if len(image_grid_thw.shape) > 1:
                result["image_grid_thw"] = image_grid_thw[0]
            else:
                result["image_grid_thw"] = image_grid_thw
        
        return result


def collate(batch):
    """Collate function for batching."""
    keys = batch[0].keys()
    out = {}
    
    for k in keys:
        if k == "image_grid_thw":
            # Stack grid_thw carefully
            if all(k in b for b in batch):
                out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    
    return out
