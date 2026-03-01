"""
Standalone Evaluation Script for Geometric Dataset
Re-evaluate inference results without running inference again.

Ollama (qwen3:8b) for semantic evaluationをサポートするように修正されています。
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys
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
  # 基本的な再評価 (全文一致と部分一致のみ)
  python run_evaluation.py --base_dir /path/to/geometric_datasets
  
  # OllamaとQwen3:8Bを使用したセマンティック評価
  python run_evaluation.py --base_dir /path/to/geometric_datasets --use_semantic --use_ollama --ollama_model qwen3:8b
  
  # 別のOllamaモデルを指定
  python run_evaluation.py --base_dir /path/to/geometric_datasets --use_semantic --use_ollama --ollama_model llama2
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
        help="LLM APIを使用してセマンティック類似性評価を実行します (OpenAI または Ollama)"
    )
    
    # --- OpenAI Arguments (互換性のために残す) ---
    parser.add_argument(
        "--openai_model",
        type=str,
        default=None,
        help="OpenAIモデル名 (例: gpt-4)。--use_ollamaが設定されていない場合のみ使用されます。"
    )

    # --- Ollama Arguments (新規追加) ---
    parser.add_argument(
        "--use_ollama",
        action="store_true",
        help="セマンティック評価にOpenAI APIの代わりにOllamaを使用します"
    )

    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen3:0.6b", # ユーザー要求のモデルをデフォルト値とする
        help="Ollamaモデル名 (default: qwen3:8b)。--use_ollamaが設定されている場合のみ使用されます。"
    )
    
    parser.add_argument(
        "--single_csv",
        type=str,
        default=None,
        help="ディレクトリ全体ではなく単一のCSVファイルを評価します"
    )
    
    args = parser.parse_args()
    
    # --- Input Validation and Setup ---
    print("\n" + "="*60)
    print("Geometric Dataset Evaluation (Standalone)")
    print("="*60)

    # LLMプロバイダーとモデル名の設定
    llm_kwargs = {}
    if args.use_semantic:
        if args.use_ollama:
            llm_kwargs['llm_provider'] = 'ollama'
            llm_kwargs['llm_model'] = args.ollama_model
            print(f"Semantic evaluation using Ollama model: {args.ollama_model}")
        elif args.openai_model:
            llm_kwargs['llm_provider'] = 'openai'
            llm_kwargs['llm_model'] = args.openai_model
            print(f"Semantic evaluation using OpenAI model: {args.openai_model}")
        else:
            print("Error: --use_semantic requires either --openai_model or --use_ollama to be set.")
            sys.exit(1)
    else:
        llm_kwargs['llm_provider'] = 'none'
        print("Running basic evaluation (exact and partial match only).")


    # --- Run Evaluation ---
    if args.single_csv:
        # 単一のCSVファイルを評価
        print(f"Evaluating single file: {args.single_csv}")
        
        # 修正: GeometricDatasetEvaluatorのインスタンス化をLLMプロバイダーのロジックに合わせて修正
        # NOTE: 外部ライブラリ (evaluation_metrics.py) のコードに依存します
        # 以下のコードは、evaluatorが新しい引数を受け入れることを想定しています。
        # エラーが出たため、引数を元の状態に戻し、use_ollamaのロジックは内部で処理されることを期待します。

        # ----------------------------------------------------
        # 暫定的な修正：llm_provider/llm_modelを直接渡すのをやめる
        # ----------------------------------------------------

        model_to_use = args.ollama_model if args.use_ollama else args.openai_model

        evaluator = GeometricDatasetEvaluator(
            use_semantic=args.use_semantic,
            openai_model=model_to_use, # Ollamaモデル名をここに渡す
            use_ollama=args.use_ollama  # 💡 追加: use_ollamaフラグを渡す
        )
        
        result = evaluator.evaluate_csv(args.single_csv)
        
        print("\n--- Evaluation Results ---")
        print(f"Scenes: {result.num_scenes}")
        print(f"QA Pairs: {result.num_qa_pairs}")
        print(f"Exact Match: {result.avg_exact_match:.4f}")
        print(f"Partial Match: {result.avg_partial_match:.4f}")
        if hasattr(result, 'avg_semantic_similarity') and result.avg_semantic_similarity is not None:
            print(f"Semantic Similarity: {result.avg_semantic_similarity:.4f}")
        
    else:
        # ディレクトリ全体を評価
        print(f"Base directory: {args.base_dir}")
        print(f"use_ollama: {args.use_ollama}")
        # run_standalone_evaluationが元の引数のみを受け入れると仮定し、
        # Ollamaモデルをopenai_model引数として渡します。
        model_to_use = args.ollama_model if args.use_ollama else args.openai_model
        
        output_paths = run_standalone_evaluation(
            base_dir=args.base_dir,
            use_semantic=args.use_semantic,
            openai_model=model_to_use, 
            # 💡 追加: run_standalone_evaluation 関数に use_ollama フラグを渡す
            use_ollama=args.use_ollama 
        )
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        print("\nOutput files:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()