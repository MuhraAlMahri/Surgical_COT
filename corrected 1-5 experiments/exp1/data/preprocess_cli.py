#!/usr/bin/env python3
"""
Command-line interface for preprocessing JSONL files.

Usage:
    python preprocess_cli.py input.jsonl output.jsonl
    python preprocess_cli.py --help
"""

import json
import re
import sys
from pathlib import Path

# Import from local schema (works when run as script)
try:
    from schema import infer_question_type, build_candidates
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from schema import infer_question_type, build_candidates


def normalize_answer(ans: str) -> str:
    """Normalize answer text for consistent evaluation."""
    x = ans.strip().lower()
    x = re.sub(r"[^\w\.\-\% ]+", "", x)  # keep simple tokens
    return x


def enrich_jsonl(in_path, out_path, verbose=True):
    """
    Enrich JSONL file with question types and answer candidates.
    
    Args:
        in_path: Input JSONL file path
        out_path: Output JSONL file path
        verbose: Print progress information
    
    Returns:
        Number of samples processed
    """
    out = []
    
    if verbose:
        print(f"Reading from: {in_path}")
    
    with open(in_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                q = ex["question"]
                gt = normalize_answer(ex["answer"])
                qtype = ex.get("question_type") or infer_question_type(q)
                
                # Enrich the example
                ex["question_type"] = qtype
                ex["answer"] = gt
                ex["answer_candidates"] = build_candidates(qtype, ex)
                
                out.append(ex)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write enriched data
    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    
    if verbose:
        print(f"Wrote {len(out)} enriched samples to: {out_path}")
        
        # Print statistics
        type_counts = {}
        for ex in out:
            qtype = ex.get("question_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("\nQuestion type distribution:")
        for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(out) * 100) if len(out) > 0 else 0
            print(f"  {qtype:15} {count:5} ({pct:5.1f}%)")
    
    return len(out)


def convert_json_to_jsonl(in_path, out_path, verbose=True):
    """
    Convert JSON array to JSONL format with enrichment.
    
    Args:
        in_path: Input JSON file (array of objects)
        out_path: Output JSONL file path
        verbose: Print progress information
    """
    if verbose:
        print(f"Converting JSON to JSONL: {in_path}")
    
    with open(in_path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of objects")
    
    # Create temporary JSONL
    temp_path = Path(out_path).with_suffix('.temp.jsonl')
    with open(temp_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    # Enrich it
    count = enrich_jsonl(str(temp_path), out_path, verbose=verbose)
    
    # Clean up temp file
    temp_path.unlink()
    
    return count


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess surgical VQA data with question type inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enrich a JSONL file
  python preprocess_cli.py train.jsonl train_enriched.jsonl
  
  # Convert JSON to enriched JSONL
  python preprocess_cli.py --from-json data.json data.jsonl
  
  # Quiet mode
  python preprocess_cli.py -q input.jsonl output.jsonl
        """
    )
    
    parser.add_argument('input', help='Input file path (JSONL or JSON)')
    parser.add_argument('output', help='Output JSONL file path')
    parser.add_argument('--from-json', action='store_true',
                       help='Input is JSON array (will convert to JSONL)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Process based on input format
    try:
        if args.from_json:
            count = convert_json_to_jsonl(args.input, args.output, verbose=not args.quiet)
        else:
            count = enrich_jsonl(args.input, args.output, verbose=not args.quiet)
        
        if not args.quiet:
            print(f"\n✓ Successfully processed {count} samples")
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

