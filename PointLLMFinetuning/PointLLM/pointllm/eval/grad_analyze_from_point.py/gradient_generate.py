"""
pointllm_gradient_attribution.py

PointLLMモデルの予測に対する各点群ポイントの影響度を勾配ベースで計算します。
入力点群(8192点)と正解ラベルから、各点が損失にどれだけ寄与しているかを分析します。

主な機能:
- 点群の各座標(x,y,z)とRGB値に対する勾配計算
- 損失関数に基づく影響度スコアの算出
- 複数の集約方法(L2ノルム、絶対値和、最大値など)をサポート

使用例:
    from pointllm_gradient_attribution import compute_point_importance
    
    importance_scores = compute_point_importance(
        model=model,
        tokenizer=tokenizer,
        point_cloud=point_cloud_tensor,  # shape: [1, 8192, 6]
        question="この歯科モデルの特徴は?",
        target_text="これは下顎の歯科モデルです。",
        device=device,
        aggregation="l2_norm"  # or "abs_sum", "max_abs", "mean_abs"
    )
    # importance_scores: shape [8192], 各点の重要度スコア
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Literal
import warnings


class PointCloudGradientAttribution:
    """点群に対する勾配ベースの帰属分析クラス"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            model: PointLLMモデル
            tokenizer: トークナイザー
            device: 計算デバイス
            dtype: 計算精度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
        # モデルを評価モードに
        self.model.eval()
        
        # Point Backbone Configの取得
        self.point_config = self._get_point_config()
        
    def _get_point_config(self) -> dict:
        """モデルからpoint_backbone_configを取得"""
        if hasattr(self.model, "get_model"):
            return self.model.get_model().point_backbone_config
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_model"):
            return self.model.base_model.get_model().point_backbone_config
        else:
            point_config = getattr(self.model.config, "point_backbone_config", None)
            if point_config is None:
                point_config = self.model.base_model.model.point_backbone_config
            return point_config
    
    def _clean_question_text(self, content: str) -> str:
        """質問文から不正な特殊トークンを除去"""
        import re
        
        # 誤った形式のパターンを削除
        patterns = [
            r'<point_start><point_patch>\*(\d+)<point_end>',
            r'<point><point_patch>\*(\d+)',
            r'<point_patch>\*(\d+)'
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content)
        
        # 特殊トークンを削除
        for token in ['<point_start>', '<point_end>', '<point_patch>', '<point>']:
            content = content.replace(token, '')
        
        return content.strip()
    
    def _prepare_prompt(self, question: str) -> str:
        """プロンプトを構築"""
        from pointllm.conversation import conv_templates
        
        question = self._clean_question_text(question)
        
        point_token_len = self.point_config['point_token_len']
        default_point_patch_token = self.point_config['default_point_patch_token']
        mm_use_point_start_end = self.point_config.get('mm_use_point_start_end', False)
        
        # プロンプト構築
        if mm_use_point_start_end:
            start = self.point_config['default_point_start_token']
            end = self.point_config['default_point_end_token']
            prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
        else:
            prompt = f"{default_point_patch_token * point_token_len}\n{question}"
        
        # Conversation Template
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        
        return full_prompt
    
    def _tokenize_target(self, target_text: str) -> torch.Tensor:
        """正解テキストをトークン化"""
        tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        return tokens.input_ids.to(self.device)
    
    def compute_loss(
        self,
        point_cloud: torch.Tensor,
        question: str,
        target_text: str,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        モデルの損失を計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6]
            question: 質問文
            target_text: 正解テキスト
            return_logits: logitsも返すかどうか
            
        Returns:
            loss: 損失値
            logits: (オプション) モデルの出力logits
        """
        # プロンプト準備
        full_prompt = self._prepare_prompt(question)
        
        # 入力のトークン化
        inputs = self.tokenizer([full_prompt], return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        # 正解のトークン化
        target_ids = self._tokenize_target(target_text)
        
        # ラベルの構築: 入力部分は-100(無視)、正解部分のみ学習対象
        labels = torch.full_like(input_ids, -100)
        labels = torch.cat([labels, target_ids], dim=1)
        
        # 入力IDも正解分を結合
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        
        # モデルのフォワード
        outputs = self.model(
            input_ids=full_input_ids,
            point_clouds=point_cloud,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        if return_logits:
            return loss, outputs.logits
        return loss, None
    
    def compute_gradients(
        self,
        point_cloud: torch.Tensor,
        question: str,
        target_text: str
    ) -> torch.Tensor:
        """
        点群の各要素に対する勾配を計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6], requires_grad=False
            question: 質問文
            target_text: 正解テキスト
            
        Returns:
            gradients: 勾配テンソル [N, 6]
        """
        # 点群のコピーを作成し、勾配計算を有効化
        pc_copy = point_cloud.clone().detach().to(dtype=self.dtype)
        pc_copy.requires_grad = True
        
        # 損失計算
        loss, _ = self.compute_loss(pc_copy, question, target_text)
        
        # 勾配計算
        loss.backward()
        
        # 勾配を取得 [1, N, 6] -> [N, 6]
        gradients = pc_copy.grad.squeeze(0).detach()
        
        return gradients
    
    def compute_importance_scores(
        self,
        gradients: torch.Tensor,
        point_cloud: torch.Tensor,
        aggregation: Literal["l2_norm", "abs_sum", "max_abs", "mean_abs", "gradient_x_input"] = "l2_norm"
    ) -> np.ndarray:
        """
        勾配から各点の重要度スコアを計算
        
        Args:
            gradients: 勾配テンソル [N, 6]
            point_cloud: 元の点群 [1, N, 6]
            aggregation: 集約方法
                - "l2_norm": L2ノルム (デフォルト)
                - "abs_sum": 絶対値の合計
                - "max_abs": 絶対値の最大値
                - "mean_abs": 絶対値の平均
                - "gradient_x_input": 勾配×入力 (Integrated Gradients風)
                
        Returns:
            importance_scores: 各点の重要度 [N]
        """
        # 点群を [N, 6] に整形
        pc = point_cloud.squeeze(0)
        
        if aggregation == "l2_norm":
            # 各点の6次元ベクトルのL2ノルム
            scores = torch.norm(gradients, p=2, dim=1)
            
        elif aggregation == "abs_sum":
            # 絶対値の合計
            scores = torch.sum(torch.abs(gradients), dim=1)
            
        elif aggregation == "max_abs":
            # 絶対値の最大値
            scores = torch.max(torch.abs(gradients), dim=1)[0]
            
        elif aggregation == "mean_abs":
            # 絶対値の平均
            scores = torch.mean(torch.abs(gradients), dim=1)
            
        elif aggregation == "gradient_x_input":
            # 勾配×入力値（Integrated Gradients的なアプローチ）
            scores = torch.sum(gradients * pc, dim=1).abs()
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return scores.cpu().numpy()


def compute_point_importance(
    model,
    tokenizer,
    point_cloud: torch.Tensor,
    question: str,
    target_text: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    aggregation: Literal["l2_norm", "abs_sum", "max_abs", "mean_abs", "gradient_x_input"] = "l2_norm",
    normalize: bool = True
) -> np.ndarray:
    """
    点群の各点が損失に与える影響度を計算
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群テンソル [1, 8192, 6]
        question: 質問文
        target_text: 正解テキスト
        device: 計算デバイス
        dtype: 計算精度
        aggregation: スコア集約方法
        normalize: スコアを[0,1]に正規化するか
        
    Returns:
        importance_scores: 各点の重要度スコア [8192]
    """
    # Attribution計算器の初期化
    attributor = PointCloudGradientAttribution(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )
    
    # 点群をデバイスとdtypeに変換
    point_cloud = point_cloud.to(device=device, dtype=dtype)
    
    # 勾配計算
    print(f"Computing gradients for {point_cloud.shape[1]} points...")
    gradients = attributor.compute_gradients(point_cloud, question, target_text)
    
    # 重要度スコア計算
    print(f"Computing importance scores (aggregation: {aggregation})...")
    scores = attributor.compute_importance_scores(
        gradients, 
        point_cloud, 
        aggregation=aggregation
    )
    
    # 正規化
    if normalize and scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    return scores


def visualize_importance(
    point_cloud: np.ndarray,
    importance_scores: np.ndarray,
    save_path: Optional[str] = None,
    colormap: str = "hot"
) -> None:
    """
    重要度スコアを可視化（オプション）
    
    Args:
        point_cloud: 点群 [N, 6] (xyz + rgb)
        importance_scores: 重要度スコア [N]
        save_path: 保存パス
        colormap: カラーマップ名
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        
        # 正規化
        norm_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
        
        # カラーマップ適用
        cmap = cm.get_cmap(colormap)
        colors = cmap(norm_scores)
        
        # 3Dプロット
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        xyz = point_cloud[:, :3]
        scatter = ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=colors,
            s=1,
            alpha=0.6
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Importance Visualization')
        
        # カラーバー
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(norm_scores)
        plt.colorbar(sm, ax=ax, label='Importance Score')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        warnings.warn("matplotlib is not installed. Skipping visualization.")


def analyze_importance_distribution(
    importance_scores: np.ndarray,
    percentiles: list = [50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """
    重要度スコアの統計情報を分析
    
    Args:
        importance_scores: 重要度スコア [N]
        percentiles: 計算するパーセンタイル
        
    Returns:
        stats: 統計情報の辞書
    """
    stats = {
        "mean": float(np.mean(importance_scores)),
        "std": float(np.std(importance_scores)),
        "min": float(np.min(importance_scores)),
        "max": float(np.max(importance_scores)),
    }
    
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(importance_scores, p))
    
    return stats


# =============================================================================
# 使用例
# =============================================================================

if __name__ == "__main__":
    """
    使用例: 勾配ベースの点群重要度分析
    """
    import sys
    import os
    
    # PointLLM環境セットアップ（既存スクリプトから流用）
    from chat_gradio_full_llm_finetune import (
        setup_pointllm_environment,
        get_device,
        load_full_finetuned_model,
        load_and_process_point_cloud
    )
    
    # 環境セットアップ
    setup_pointllm_environment()
    device = get_device("cuda", verbose=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # モデルロード（例）
    model_path = "RunsenXu/PointLLM_7B_v1.2"
    phase1_checkpoint = "./checkpoints/phase1/checkpoint-epoch-20"
    phase2_checkpoint = "./checkpoints/phase2/best_model"
    
    print("Loading model...")
    model, tokenizer = load_full_finetuned_model(
        model_path=model_path,
        phase1_checkpoint_path=phase1_checkpoint,
        phase2_checkpoint_path=phase2_checkpoint,
        device=device,
        dtype=dtype
    )
    
    # 点群ロード
    pc_path = "./data/point_clouds/sample.npy"
    point_cloud = load_and_process_point_cloud(pc_path, num_points=8192)
    
    # 質問と正解
    question = "この歯科モデルの特徴を説明してください。"
    target_text = "これは下顎の歯科CBCTモデルです。全ての臼歯が確認できます。"
    
    # 重要度計算
    importance_scores = compute_point_importance(
        model=model,
        tokenizer=tokenizer,
        point_cloud=point_cloud,
        question=question,
        target_text=target_text,
        device=device,
        dtype=dtype,
        aggregation="l2_norm",
        normalize=True
    )
    
    print(f"\n{'='*60}")
    print("Importance Score Analysis")
    print(f"{'='*60}")
    
    # 統計情報
    stats = analyze_importance_distribution(importance_scores)
    for key, value in stats.items():
        print(f"{key:10s}: {value:.6f}")
    
    # Top-K重要点の表示
    top_k = 10
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    
    print(f"\nTop-{top_k} Most Important Points:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  Rank {rank}: Point {idx:5d} | Score: {importance_scores[idx]:.6f}")
    
    # 可視化（オプション）
    visualize_importance(
        point_cloud=point_cloud.squeeze(0).cpu().numpy(),
        importance_scores=importance_scores,
        save_path="importance_visualization.png"
    )
    
    # 結果保存
    np.save("importance_scores.npy", importance_scores)
    print(f"\n✅ Importance scores saved to: importance_scores.npy")