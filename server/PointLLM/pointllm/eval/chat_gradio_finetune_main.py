"""
main.py - Batch Inference and Evaluation for PointLLM Test Dataset

test.jsonから全てのデータを読み込んで推論を実行し、
生成結果と正解をExcel/CSVに保存します。

使用方法:
    python main.py \
        --dataset_path "path/to/llm_dataset" \
        --test_json "annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --checkpoint_path "path/to/point_proj.pt" \
        --output_file "test_results.xlsx" \
        --device "cuda"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# chat_gradio_finetuning.pyのモジュールをインポート
from chat_gradio_finetuning import (
    setup_pointllm_environment,
    load_model_and_checkpoint,
    load_and_process_point_cloud,
    generate_response,
    get_device
)


def extract_question_from_conversation(conversation: List[Dict[str, str]]) -> str:
    """
    会話データから質問文を抽出します（<point_*>トークンを除く）
    
    Args:
        conversation: 会話データのリスト
    
    Returns:
        str: 質問文（トークンを除いた部分）
    """
    for turn in conversation:
        if turn["role"] == "user":
            content = turn["content"]
            # <point_*>トークンを除去
            lines = content.split('\n')
            # 最初の行はトークンなのでスキップ
            if len(lines) > 1:
                return '\n'.join(lines[1:])
            return content
    return ""


def extract_ground_truth(conversation: List[Dict[str, str]]) -> str:
    """
    会話データから正解の回答を抽出します
    
    Args:
        conversation: 会話データのリスト
    
    Returns:
        str: 正解の回答
    """
    for turn in conversation:
        if turn["role"] == "assistant":
            return turn["content"]
    return ""


def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """
    テストデータセットを読み込みます
    
    Args:
        dataset_path: データセットのルートパス
        test_json: test.jsonの相対パス
    
    Returns:
        Dict: テストデータセット
    """
    test_json_path = os.path.join(dataset_path, test_json)
    
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON not found: {test_json_path}")
    
    print(f"\n{'='*80}")
    print(f"Loading Test Dataset")
    print(f"{'='*80}")
    print(f"Dataset path: {dataset_path}")
    print(f"Test JSON: {test_json_path}")
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    num_samples = len(test_data['data'])
    print(f"✅ Loaded {num_samples} test samples")
    print(f"{'='*80}\n")
    
    return test_data


def process_single_sample(
    sample: Dict[str, Any],
    dataset_path: str,
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    単一サンプルの推論を実行します
    
    Args:
        sample: サンプルデータ
        dataset_path: データセットのルートパス
        model: PointLLMモデル
        tokenizer: トークナイザー
        device: デバイス
        args: コマンドライン引数
    
    Returns:
        Dict: 推論結果を含む辞書
    """
    sample_id = sample['id']
    point_cloud_path = os.path.join(dataset_path, sample['point_cloud'])
    conversations = sample['conversations']
    metadata = sample.get('metadata', {})
    
    # 質問と正解を抽出
    question = extract_question_from_conversation(conversations)
    ground_truth = extract_ground_truth(conversations)
    
    # 点群の読み込み
    try:
        point_cloud = load_and_process_point_cloud(
            point_cloud_path=point_cloud_path,
            num_points=args.num_points
        )
    except Exception as e:
        print(f"❌ Error loading point cloud: {e}")
        return {
            'id': sample_id,
            'point_cloud': sample['point_cloud'],
            'question': question,
            'ground_truth': ground_truth,
            'generated': f"ERROR: {str(e)}",
            'metadata': metadata,
            'error': str(e)
        }
    
    # 推論の実行
    try:
        generated = generate_response(
            model=model,
            tokenizer=tokenizer,
            point_cloud=point_cloud,
            question=question,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        generated = f"ERROR: {str(e)}"
    
    # 結果を返す
    result = {
        'id': sample_id,
        'point_cloud': sample['point_cloud'],
        'question': question,
        'ground_truth': ground_truth,
        'generated': generated,
        'metadata': metadata
    }
    
    return result


def run_batch_inference(
    test_data: Dict[str, Any],
    dataset_path: str,
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """
    バッチ推論を実行します
    
    Args:
        test_data: テストデータセット
        dataset_path: データセットのルートパス
        model: PointLLMモデル
        tokenizer: トークナイザー
        device: デバイス
        args: コマンドライン引数
    
    Returns:
        List[Dict]: 推論結果のリスト
    """
    results = []
    samples = test_data['data']
    
    print(f"\n{'='*80}")
    print(f"Running Batch Inference")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(samples)} - ID: {sample['id']}")
        print(f"{'='*80}")
        
        result = process_single_sample(
            sample=sample,
            dataset_path=dataset_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args
        )
        
        results.append(result)
        
        # 中間結果の表示
        print(f"\n--- Result ---")
        print(f"Question: {result['question'][:100]}...")
        print(f"Ground Truth: {result['ground_truth'][:100]}...")
        print(f"Generated: {result['generated'][:100]}...")
        print(f"{'='*80}\n")
        
        # 定期的に保存（オプション）
        if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
            save_results_to_file(results, args.output_file, format=args.output_format)
            print(f"💾 Intermediate results saved at sample {i+1}")
    
    return results


def save_results_to_file(
    results: List[Dict[str, Any]],
    output_file: str,
    format: str = "xlsx"
):
    """
    結果をファイルに保存します（修正版：None判定を強化）
    
    Args:
        results: 推論結果のリスト
        output_file: 出力ファイルパス
        format: 出力フォーマット ("xlsx" or "csv")
    """
    # DataFrameに変換
    df_data = []
    for result in results:
        metadata = result.get('metadata', {})
        
        # primary_region と secondary_region が None の場合でも空辞書 {} を使うように安全策を追加
        # (metadata.get('key') or {}) という書き方は、値が None の場合も {} に変換します
        primary_region = metadata.get('primary_region') or {}
        secondary_region = metadata.get('secondary_region') or {}
        
        row = {
            'ID': result['id'],
            'Point Cloud': result['point_cloud'],
            'Question': result['question'],
            'Ground Truth': result['ground_truth'],
            'Generated': result['generated'],
            'Error': result.get('error', ''),
            
            # メタデータ
            'Tracking Source': metadata.get('tracking_source', ''),
            'Colorization Effect': metadata.get('colorization_effect', ''),
            'Variation Index': metadata.get('variation_index', ''),
            'Hand Distance (mm)': metadata.get('hand_distance_mm', ''),
            
            # ここを修正: 事前に取得した変数(辞書)を使用
            'Primary Region': primary_region.get('name', ''),
            'Primary Confidence': primary_region.get('confidence', ''),
            'Secondary Region': secondary_region.get('name', ''),
            'Secondary Confidence': secondary_region.get('confidence', ''),
        }
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # ファイル形式に応じて保存
    if format.lower() == "xlsx":
        # Excelファイルとして保存
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Test Results')
            
            # 列幅の自動調整
            worksheet = writer.sheets['Test Results']
            for column in df:
                # 列幅計算時のエラー防止（全てNaNの場合など）
                try:
                    max_len = df[column].astype(str).map(len).max()
                    column_length = max(max_len, len(str(column)))
                    col_idx = df.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_length + 2, 50)
                except:
                    pass # 幅調整に失敗しても保存は継続
        
        print(f"\n✅ Results saved to: {output_file}")
    
    elif format.lower() == "csv":
        # CSVファイルとして保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Results saved to: {output_file}")
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # 統計情報の表示
    print(f"\n{'='*80}")
    print("Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    errors = sum(1 for r in results if 'error' in r)
    print(f"Successful: {len(results) - errors}")
    print(f"Errors: {errors}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference on PointLLM test dataset"
    )
    
    # データセット設定
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="データセットのルートパス"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="annotations/test.json",
        help="test.jsonの相対パス"
    )
    
    # モデル設定
    parser.add_argument(
        "--model_path",
        type=str,
        default="RunsenXu/PointLLM_7B_v1.2",
        help="ベースモデルのパス"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="ファインチューニングされたPoint Projectorのチェックポイントパス"
    )
    parser.add_argument(
        "--pointllm_path",
        type=str,
        default=None,
        help="PointLLMリポジトリのパス（自動検出されない場合）"
    )
    
    # デバイス設定
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="使用するデバイス"
    )
    
    # 生成設定
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=8192)
    
    # 出力設定
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_results.xlsx",
        help="出力ファイルパス"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["xlsx", "csv"],
        default="xlsx",
        help="出力フォーマット"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="中間保存の間隔（0で無効化）"
    )
    
    args = parser.parse_args()
    
    # タイムスタンプを追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.output_file.endswith(('.xlsx', '.csv')):
        args.output_file = f"{args.output_file}_{timestamp}.{args.output_format}"
    
    print(f"\n{'='*80}")
    print("PointLLM Batch Inference")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Test JSON: {args.test_json}")
    print(f"Model: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {args.output_file}")
    print(f"{'='*80}\n")
    
    # デバイスの設定
    if args.device == "auto":
        device = get_device(verbose=True)
    else:
        device = torch.device(args.device)
        print(f"[INFO] Using specified device: {device}")
    
    # データ型の設定
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    print(f"[INFO] Using dtype: {dtype}\n")
    
    # PointLLM環境のセットアップ
    if not setup_pointllm_environment(args.pointllm_path):
        print("❌ Failed to setup PointLLM environment")
        return
    
    # モデルの読み込み
    model, tokenizer = load_model_and_checkpoint(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype
    )
    
    # テストデータセットの読み込み
    test_data = load_test_dataset(
        dataset_path=args.dataset_path,
        test_json=args.test_json
    )
    
    # バッチ推論の実行
    results = run_batch_inference(
        test_data=test_data,
        dataset_path=args.dataset_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args
    )
    
    # 結果の保存
    save_results_to_file(
        results=results,
        output_file=args.output_file,
        format=args.output_format
    )
    
    print(f"\n{'='*80}")
    print("Batch Inference Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
