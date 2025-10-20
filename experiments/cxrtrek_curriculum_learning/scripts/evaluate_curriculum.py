#!/usr/bin/env python3
"""
Evaluate Curriculum Learning Model

Tests the final curriculum learning model (stage3_best) on all three stages
and compares performance to CXRTrek Sequential baseline.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class EvaluationDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, data_path: str, image_dir: str, stage_num: int = None):
        """
        Args:
            data_path: Path to JSON data file
            image_dir: Directory containing images
            stage_num: Which stage to extract (1, 2, 3, or None for all)
        """
        self.image_dir = image_dir
        self.stage_num = stage_num
        
        # Load data
        with open(data_path, 'r') as f:
            all_data = json.load(f)
        
        # Extract QA pairs
        self.samples = []
        for item in all_data:
            image_path = item.get('image_path', item.get('image', ''))
            
            # Handle image path
            if not os.path.isabs(image_path):
                if image_path.startswith('images/'):
                    image_path = image_path[7:]
                image_path = os.path.join(image_dir, image_path)
            
            # Extract QA pairs
            stages_data = item.get('stages', item.get('clinical_flow_stages', {}))
            
            if stage_num:
                # Extract specific stage
                qa_pairs = self._extract_stage_qa(stages_data, stage_num)
                for qa in qa_pairs:
                    self.samples.append({
                        'image_path': image_path,
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'stage': stage_num,
                        'image_id': item.get('image_id', '')
                    })
            else:
                # Extract all stages
                for s in [1, 2, 3]:
                    qa_pairs = self._extract_stage_qa(stages_data, s)
                    for qa in qa_pairs:
                        self.samples.append({
                            'image_path': image_path,
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'stage': s,
                            'image_id': item.get('image_id', '')
                        })
        
        # Use test split (last 10%)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(0.9 * len(self.samples))
        self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        print(f"Loaded {len(self.samples)} test samples")
    
    def _extract_stage_qa(self, stages_data: Dict, stage_num: int) -> List[Dict]:
        """Extract QA pairs for a specific stage."""
        possible_keys = [
            f'stage_{stage_num}',
            f'Stage-{stage_num}',
            f'Stage {stage_num}',
            f'Stage-{stage_num}: Initial Assessment',
            f'Stage-{stage_num}: Findings Identification',
            f'Stage-{stage_num}: Clinical Context'
        ]
        
        for key in possible_keys:
            if key in stages_data:
                return stages_data[key]
        
        return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, processor):
    """Collate function for batching."""
    images = []
    questions = []
    answers = []
    stages = []
    
    for item in batch:
        image = Image.open(item['image_path']).convert('RGB')
        images.append(image)
        questions.append(item['question'])
        answers.append(item['answer'])
        stages.append(item['stage'])
    
    # Prepare messages
    messages_batch = []
    for q in questions:
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q}
            ]
        }])
    
    # Process
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
             for msg in messages_batch]
    
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    return inputs, answers, stages


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate accuracy with flexible matching."""
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.lower().strip()
        gt_clean = gt.lower().strip()
        
        # Exact match
        if pred_clean == gt_clean:
            correct += 1
        # Partial match (prediction contains ground truth or vice versa)
        elif gt_clean in pred_clean or pred_clean in gt_clean:
            correct += 1
        # Yes/No matching
        elif ('yes' in pred_clean and 'yes' in gt_clean) or \
             ('no' in pred_clean and 'no' in gt_clean):
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def evaluate_model(
    model,
    processor,
    dataloader,
    device: str = "cuda"
):
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_stages = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch_inputs, batch_answers, batch_stages in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            # Generate
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
            # Decode predictions
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract answer portion (after the question)
            cleaned_predictions = []
            for pred in predictions:
                # Remove the input prompt
                if "assistant\n" in pred:
                    pred = pred.split("assistant\n")[-1]
                cleaned_predictions.append(pred.strip())
            
            all_predictions.extend(cleaned_predictions)
            all_ground_truths.extend(batch_answers)
            all_stages.extend(batch_stages)
    
    return all_predictions, all_ground_truths, all_stages


def main():
    parser = argparse.ArgumentParser(description="Evaluate Curriculum Learning Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to data JSON")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output", type=str, default="curriculum_results.json", help="Output file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Set device early
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cpu"
    
    print("="*60)
    print("CURRICULUM LEARNING EVALUATION")
    print("="*60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    # Load model
    print("Loading model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    print("✓ Model loaded")
    
    # Create dataset
    print("\nCreating test dataset...")
    dataset = EvaluationDataset(args.data, args.images)
    
    from functools import partial
    collate_fn_with_processor = partial(collate_fn, processor=processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_processor,
        num_workers=0
    )
    
    # Evaluate
    predictions, ground_truths, stages = evaluate_model(
        model, processor, dataloader, device
    )
    
    # Calculate overall accuracy
    overall_acc = calculate_accuracy(predictions, ground_truths)
    
    # Calculate per-stage accuracy
    stage_results = {}
    for stage_num in [1, 2, 3]:
        stage_preds = [p for p, s in zip(predictions, stages) if s == stage_num]
        stage_gts = [g for g, s in zip(ground_truths, stages) if s == stage_num]
        
        if stage_preds:
            stage_acc = calculate_accuracy(stage_preds, stage_gts)
            stage_results[f"stage_{stage_num}"] = {
                "accuracy": stage_acc,
                "num_samples": len(stage_preds)
            }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
    print(f"Total Samples: {len(predictions)}")
    
    print("\nPer-Stage Results:")
    for stage_num in [1, 2, 3]:
        key = f"stage_{stage_num}"
        if key in stage_results:
            acc = stage_results[key]["accuracy"]
            num = stage_results[key]["num_samples"]
            print(f"  Stage {stage_num}: {acc*100:.2f}% ({num} samples)")
    
    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "overall_accuracy": overall_acc,
        "total_samples": len(predictions),
        "stage_results": stage_results,
        "predictions": [
            {
                "prediction": p,
                "ground_truth": g,
                "stage": s
            }
            for p, g, s in zip(predictions, ground_truths, stages)
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
