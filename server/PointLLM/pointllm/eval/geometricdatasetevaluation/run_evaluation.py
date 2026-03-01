"""
Standalone Evaluation Script for Geometric Dataset
Re-evaluate inference results without running inference again.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_metrics import (
    GeometricDatasetEvaluator,
    EvaluationReportGenerator,
    run_standalone_evaluation
)


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate Geometric Dataset inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic re-evaluation (exact match and partial match only)
  python run_evaluation.py --base_dir /path/to/geometric_datasets
  
  # With semantic evaluation using GPT API
  python run_evaluation.py --base_dir /path/to/geometric_datasets --use_semantic
  
  # Specify different OpenAI model
  python run_evaluation.py --base_dir /path/to/geometric_datasets --use_semantic --openai_model gpt-4
        """
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing geometric_datasets with inference results"
    )
    
    parser.add_argument(
        "--use_semantic",
        action="store_true",
        help="Use GPT API for semantic similarity evaluation"
    )
    
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model for semantic evaluation (default: gpt-4.1-nano)"
    )
    
    parser.add_argument(
        "--single_csv",
        type=str,
        default=None,
        help="Evaluate a single CSV file instead of entire directory"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Geometric Dataset Evaluation (Standalone)")
    print("="*60)
    
    if args.single_csv:
        # Evaluate single CSV file
        print(f"Evaluating single file: {args.single_csv}")
        
        evaluator = GeometricDatasetEvaluator(
            use_semantic=args.use_semantic,
            openai_model=args.openai_model
        )
        
        result = evaluator.evaluate_csv(args.single_csv)
        
        print("\n--- Evaluation Results ---")
        print(f"Scenes: {result.num_scenes}")
        print(f"QA Pairs: {result.num_qa_pairs}")
        print(f"Exact Match: {result.avg_exact_match:.4f}")
        print(f"Partial Match: {result.avg_partial_match:.4f}")
        if result.avg_semantic_similarity is not None:
            print(f"Semantic Similarity: {result.avg_semantic_similarity:.4f}")
        
    else:
        # Evaluate entire directory
        print(f"Base directory: {args.base_dir}")
        print(f"Semantic evaluation: {args.use_semantic}")
        if args.use_semantic:
            print(f"OpenAI model: {args.openai_model}")
        
        output_paths = run_standalone_evaluation(
            base_dir=args.base_dir,
            use_semantic=args.use_semantic,
            openai_model=args.openai_model
        )
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        print("\nOutput files:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
