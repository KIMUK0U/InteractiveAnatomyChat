"""
evaluation.py - PointLLM Evaluation and Inference Utilities

このモジュールは、学習済みPoint Projectorの評価と推論を行います。

主な機能:
- 検証データでの定量評価
- テキスト生成による定性評価
- 推論ユーティリティ
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import json

from config import FullConfig, GenerationConfig
from dataset import PointCloudProcessor


class PointLLMInference:
    """
    学習済みPointLLMモデルで推論を行うクラス
    
    Point Projectorをファインチューニングした後のモデルで、
    点群からテキスト生成を行います。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        processor: PointCloudProcessor,
        generation_config: GenerationConfig,
        device: str = "cuda"
    ):
        """
        推論クラスを初期化します
        
        Args:
            model: PointLLMモデル
            tokenizer: トークナイザー
            processor: 点群前処理オブジェクト
            generation_config: 生成設定
            device: 使用するデバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_config = generation_config
        self.device = device
        
        # 評価モードに設定
        self.model.eval()
    
    def _prepare_point_cloud(self, point_cloud: np.ndarray) -> torch.Tensor:
        """
        点群を推論用に準備します
        
        Args:
            point_cloud: 入力点群 (N, 6) または ファイルパス
        
        Returns:
            処理済み点群テンソル (1, num_points, 6)
        """
        # 文字列の場合はファイルから読み込み
        if isinstance(point_cloud, str):
            point_cloud = np.load(point_cloud)
        
        # 前処理（training=Falseでデータ拡張なし）
        processed = self.processor.process(point_cloud, training=False)
        
        # バッチ次元を追加してテンソルに変換
        tensor = torch.from_numpy(processed).unsqueeze(0)
        
        # ★ 修正箇所: モデルのdtypeに合わせてキャストする
        # モデルがbfloat16なら入力もbfloat16にする必要があります
        target_dtype = getattr(self.model, 'dtype', torch.float32)
        return tensor.to(self.device, dtype=torch.float32)
    
    def _prepare_prompt(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        プロンプトをトークン化します
        
        <point>トークンを含む質問をPointLLMが理解できる形式に変換します。
        
        Args:
            question: ユーザーの質問(<point>を含む可能性)
        
        Returns:
            (input_ids, attention_mask) のタプル
        """
        # ==========================================================
        # 特殊トークンの処理（二重展開を防止）
        # ==========================================================
        POINT_TOKEN = "<point>"
        POINT_START_TOKEN = "<point_start>"
        POINT_END_TOKEN = "<point_end>"
        POINT_PATCH_TOKEN = "<point_patch>"
        
        point_token_len = 513  # パッチ512 + グローバル1
        expanded_token = POINT_START_TOKEN + (POINT_PATCH_TOKEN * point_token_len) + POINT_END_TOKEN
        
        # ケース1: <point>タグがある場合 → 展開済みトークン列に置換
        if POINT_TOKEN in question:
            question = question.replace(POINT_TOKEN, expanded_token)
        
        # ケース2: 既に<point_start>と<point_end>がある場合 → そのまま（二重展開防止）
        elif POINT_START_TOKEN in question and POINT_END_TOKEN in question:
            pass  # 既に展開済みなので何もしない
        
        # ケース3: どちらもない場合 → プロンプトの先頭に展開済みトークン列を追加
        else:
            question = expanded_token + "\n" + question
        # ==========================================================
        
        # フォーマットの適用
        formatted_prompt = f"User: {question}\nAssistant: "
        
        # トークン化
        encoding = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
    
    @torch.no_grad()
    def generate(
        self,
        point_cloud: np.ndarray,
        question: str,
        **kwargs
    ) -> str:
        """
        点群と質問からテキストを生成します
        
        Args:
            point_cloud: 入力点群 (N, 6) または .npyファイルパス
            question: ユーザーの質問
            **kwargs: 生成パラメータのオーバーライド
        
        Returns:
            生成されたテキスト
        """
        # 点群の準備
        point_cloud_tensor = self._prepare_point_cloud(point_cloud)
        
        # プロンプトの準備
        input_ids, attention_mask = self._prepare_prompt(question)
        
        # 生成パラメータの設定
        gen_config = self.generation_config
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', gen_config.max_new_tokens),
            'do_sample': kwargs.get('do_sample', gen_config.do_sample),
            'temperature': kwargs.get('temperature', gen_config.temperature),
            'top_p': kwargs.get('top_p', gen_config.top_p),
            'top_k': kwargs.get('top_k', gen_config.top_k),
            'repetition_penalty': kwargs.get('repetition_penalty', gen_config.repetition_penalty),
            'length_penalty': kwargs.get('length_penalty', gen_config.length_penalty),
            'min_new_tokens': kwargs.get('min_new_tokens', gen_config.min_new_tokens),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # 生成
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_cloud_tensor,
            **generation_kwargs
        )
        
        # デコード
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],  # プロンプト部分を除去
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    @torch.no_grad()
    def batch_generate(
        self,
        samples: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        複数のサンプルに対してバッチ推論を行います
        
        Args:
            samples: サンプルのリスト
                    各サンプルは {'point_cloud': path, 'question': str} の形式
            show_progress: プログレスバーを表示するか
        
        Returns:
            生成結果のリスト
                各結果は {'question': str, 'generated': str, 'point_cloud': path} の形式
        """
        results = []
        
        iterator = samples
        if show_progress:
            iterator = tqdm(samples, desc="Generating")
        
        for sample in iterator:
            point_cloud = sample['point_cloud']
            question = sample['question']
            
            try:
                generated = self.generate(point_cloud, question)
                results.append({
                    'point_cloud': str(point_cloud),
                    'question': question,
                    'generated': generated,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'point_cloud': str(point_cloud),
                    'question': question,
                    'generated': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results


class Evaluator:
    """
    学習済みモデルの評価を行うクラス
    
    定量的評価（損失、精度など）と定性的評価（生成サンプル）を
    両方サポートしています。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: FullConfig,
        device: str = "cuda"
    ):
        """
        評価クラスを初期化します
        
        Args:
            model: PointLLMモデル
            tokenizer: トークナイザー
            config: 全体設定
            device: 使用するデバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 推論クラスの初期化
        self.processor = PointCloudProcessor(config.data)
        self.inference = PointLLMInference(
            model=model,
            tokenizer=tokenizer,
            processor=self.processor,
            generation_config=config.generation,
            device=device
        )
    
    @torch.no_grad()
    def compute_perplexity(self, dataloader) -> float:
        """
        パープレキシティを計算します
        
        パープレキシティは言語モデルの性能を測る指標で、
        値が低いほど良いモデルを意味します。
        
        Args:
            dataloader: 評価用DataLoader
        
        Returns:
            パープレキシティの値
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
    
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                # データをGPUに移動
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                point_clouds = batch['point_clouds'].to(self.device)
                
                # ★ 点群データを明示的にbfloat16に変換
                if self.model.dtype == torch.bfloat16:
                    point_clouds = point_clouds.to(torch.bfloat16)
                
                # フォワードパス
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    point_clouds=point_clouds,
                    labels=labels,
                )
                
                # 損失の累積
                # -100でマスクされていないトークン数をカウント
                num_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
        
        # パープレキシティの計算
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_generation_quality(
        self,
        test_samples: List[Dict],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        生成品質の定性評価を行います
        
        いくつかのサンプルに対して生成を行い、
        結果を人間が確認できる形式で返します。
        
        Args:
            test_samples: テストサンプルのリスト
            num_samples: 評価するサンプル数
        
        Returns:
            評価結果の辞書
        """
        # サンプル数を制限
        samples_to_evaluate = test_samples[:num_samples]
        
        results = self.inference.batch_generate(samples_to_evaluate)
        
        # 成功率の計算
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        
        # 結果のフォーマット
        evaluation = {
            'num_samples': len(results),
            'success_rate': success_rate,
            'samples': results
        }
        
        return evaluation
    
    def run_full_evaluation(
        self,
        val_dataloader,
        test_samples: Optional[List[Dict]] = None,
        num_qualitative_samples: int = 5
    ) -> Dict[str, Any]:
        """
        完全な評価を実行します
        
        定量評価（パープレキシティ）と定性評価（生成サンプル）の
        両方を実行します。
        
        Args:
            val_dataloader: 検証用DataLoader
            test_samples: 定性評価用のサンプル（オプション）
            num_qualitative_samples: 定性評価のサンプル数
        
        Returns:
            評価結果の辞書
        """
        print("\n" + "=" * 60)
        print("🔍 Running Full Evaluation")
        print("=" * 60 + "\n")
        
        results = {}
        
        # 1. パープレキシティの計算
        print("Computing perplexity...")
        perplexity = self.compute_perplexity(val_dataloader)
        results['perplexity'] = perplexity
        print(f"Perplexity: {perplexity:.4f}")
        
        # 2. 定性評価
        if test_samples:
            print("\nGenerating qualitative samples...")
            qualitative = self.evaluate_generation_quality(
                test_samples,
                num_samples=num_qualitative_samples
            )
            results['qualitative'] = qualitative
            
            # サンプル出力の表示
            print("\n--- Sample Generations ---")
            for i, sample in enumerate(qualitative['samples'][:3]):
                print(f"\n[Sample {i + 1}]")
                print(f"Question: {sample['question']}")
                print(f"Generated: {sample['generated']}")
        
        print("\n" + "=" * 60)
        print("✅ Evaluation Complete")
        print("=" * 60 + "\n")
        
        return results
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        評価結果をJSONファイルに保存します
        
        Args:
            results: 評価結果の辞書
            output_path: 出力ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Evaluation results saved to: {output_path}")


def quick_inference(
    model: nn.Module,
    tokenizer: Any,
    point_cloud_path: str,
    question: str,
    config: FullConfig,
    device: str = "cuda"
) -> str:
    """
    クイック推論を行うヘルパー関数
    
    単一の点群と質問に対して素早く推論を行います。
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud_path: 点群ファイルのパス
        question: 質問文
        config: 設定
        device: デバイス
    
    Returns:
        生成されたテキスト
    """
    processor = PointCloudProcessor(config.data)
    
    inference = PointLLMInference(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        generation_config=config.generation,
        device=device
    )
    
    return inference.generate(point_cloud_path, question)


def compare_before_after(
    original_model: nn.Module,
    finetuned_model: nn.Module,
    tokenizer: Any,
    point_cloud_path: str,
    question: str,
    config: FullConfig,
    device: str = "cuda"
) -> Dict[str, str]:
    """
    ファインチューニング前後の出力を比較します
    
    Args:
        original_model: オリジナルモデル
        finetuned_model: ファインチューニング済みモデル
        tokenizer: トークナイザー
        point_cloud_path: 点群ファイルのパス
        question: 質問文
        config: 設定
        device: デバイス
    
    Returns:
        {'original': str, 'finetuned': str} の辞書
    """
    processor = PointCloudProcessor(config.data)
    
    # オリジナルモデルで生成
    original_inference = PointLLMInference(
        model=original_model,
        tokenizer=tokenizer,
        processor=processor,
        generation_config=config.generation,
        device=device
    )
    original_output = original_inference.generate(point_cloud_path, question)
    
    # ファインチューニング済みモデルで生成
    finetuned_inference = PointLLMInference(
        model=finetuned_model,
        tokenizer=tokenizer,
        processor=processor,
        generation_config=config.generation,
        device=device
    )
    finetuned_output = finetuned_inference.generate(point_cloud_path, question)
    
    return {
        'question': question,
        'original': original_output,
        'finetuned': finetuned_output
    }


# エクスポート
__all__ = [
    "PointLLMInference",
    "Evaluator",
    "quick_inference",
    "compare_before_after",
]
