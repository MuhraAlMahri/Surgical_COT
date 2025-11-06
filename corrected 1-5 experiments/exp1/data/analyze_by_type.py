#!/usr/bin/env python3
"""
Analyze Exp1 results by question type using the schema.

This script demonstrates how to use the schema to categorize questions
and compute per-type accuracy metrics.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from schema import infer_question_type, build_candidates, QuestionType

def analyze_results(results_file: str):
    """Analyze evaluation results by question type."""
    
    print("="*80)
    print("SURGICAL VQA - QUESTION TYPE ANALYSIS (EXP1)")
    print("="*80)
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    print(f"\nTotal predictions: {len(predictions):,}")
    
    # Initialize counters
    type_stats = defaultdict(lambda: {
        'count': 0,
        'correct': 0,
        'examples': []
    })
    
    # Analyze each prediction
    for pred in predictions:
        question = pred.get('question', '')
        ground_truth = pred.get('ground_truth', '')
        prediction = pred.get('prediction', '')
        correct = pred.get('correct', False)
        
        # Infer question type
        qtype = infer_question_type(question)
        
        # Update stats
        type_stats[qtype]['count'] += 1
        if correct:
            type_stats[qtype]['correct'] += 1
        
        # Store examples (max 3 per type)
        if len(type_stats[qtype]['examples']) < 3:
            type_stats[qtype]['examples'].append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction[:100] + '...' if len(prediction) > 100 else prediction,
                'correct': correct
            })
    
    # Print summary table
    print("\n" + "="*80)
    print("ACCURACY BY QUESTION TYPE")
    print("="*80)
    print(f"{'Type':<20} {'Count':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*80)
    
    # Sort by count (descending)
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    total_count = 0
    total_correct = 0
    
    for qtype, stats in sorted_types:
        count = stats['count']
        correct = stats['correct']
        accuracy = (correct / count * 100) if count > 0 else 0.0
        
        total_count += count
        total_correct += correct
        
        print(f"{qtype:<20} {count:<10,} {correct:<10,} {accuracy:<10.2f}%")
    
    print("-"*80)
    overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0.0
    print(f"{'OVERALL':<20} {total_count:<10,} {total_correct:<10,} {overall_acc:<10.2f}%")
    print("="*80)
    
    # Print examples for each type
    print("\n" + "="*80)
    print("EXAMPLE QUESTIONS BY TYPE")
    print("="*80)
    
    for qtype, stats in sorted_types:
        print(f"\n{qtype.upper()} ({stats['count']:,} questions)")
        print("-"*80)
        
        for i, example in enumerate(stats['examples'], 1):
            status = "✓" if example['correct'] else "✗"
            print(f"\nExample {i} {status}:")
            print(f"  Question: {example['question']}")
            print(f"  Ground Truth: {example['ground_truth']}")
            print(f"  Prediction: {example['prediction']}")
            
            # Show candidates if applicable
            sample = {'question': example['question']}
            candidates = build_candidates(qtype, sample)
            if candidates:
                print(f"  Valid Candidates: {candidates}")
    
    # Additional analysis: candidate coverage
    print("\n" + "="*80)
    print("CANDIDATE ANSWER ANALYSIS")
    print("="*80)
    
    for qtype in ['yes_no', 'color']:
        if qtype in type_stats:
            stats = type_stats[qtype]
            sample = {}
            candidates = build_candidates(qtype, sample)
            
            if candidates:
                print(f"\n{qtype.upper()}:")
                print(f"  Defined candidates: {candidates}")
                print(f"  Total questions: {stats['count']:,}")
                print(f"  Note: Using constrained generation with these candidates")
                print(f"        could potentially improve accuracy for this type.")
    
    return type_stats


def main():
    """Main execution."""
    # Default results file
    results_file = Path(__file__).parent.parent.parent / "results" / "exp1_evaluation_results.json"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print(f"\nUsage: python analyze_by_type.py [results_file.json]")
        sys.exit(1)
    
    print(f"Analyzing: {results_file}")
    
    # Run analysis
    type_stats = analyze_results(str(results_file))
    
    # Save detailed stats
    output_file = Path(__file__).parent / "question_type_stats.json"
    
    # Convert to JSON-serializable format
    output_data = {
        qtype: {
            'count': stats['count'],
            'correct': stats['correct'],
            'accuracy': (stats['correct'] / stats['count'] * 100) if stats['count'] > 0 else 0.0,
            'examples': stats['examples']
        }
        for qtype, stats in type_stats.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Detailed statistics saved to: {output_file}")


if __name__ == "__main__":
    main()

