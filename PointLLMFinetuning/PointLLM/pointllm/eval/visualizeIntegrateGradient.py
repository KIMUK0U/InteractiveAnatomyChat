"""
visualize_integrated_gradients.py

テストデータから1サンプルを選択し，Integrated Gradientsを可視化します．
2つの可視化モードを提供します：
1. 3D空間上の点の重要度（各点の6次元IGをL2ノルムで集約）
2. 6次元の詳細分析（xyz座標とRGB色の個別寄与）

使用方法:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python pointllm/eval/visualizeIntegrateGradient.py \
    --dataset_path "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset" \
    --test_json "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset/annotations/test.json" \
    --model_path "RunsenXu/PointLLM_7B_v1.2" \
    --phase1_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_ddp_demo_ar_410_v1/checkpoints/checkpoint-epoch-20" \
    --phase2_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/demo_410_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v4_EPLR_high_30epoch/checkpoints/best_model" \
    --sample_index 40 \
    --ig_steps 50 \
    --output_dir "ig_visualizations410" \
    --device "cuda"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python pointllm/eval/visualizeIntegrateGradient.py \
    --dataset_path "/home/yyamashita/Desktop/kkimu/test/demo_32_ar_instraction_dataset" \
    --test_json "/home/yyamashita/Desktop/kkimu/test/demo_32_ar_instraction_dataset/annotations/test.json" \
    --model_path "RunsenXu/PointLLM_7B_v1.2" \
    --phase1_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_v9_demo_dental_model_32_final/checkpoints/checkpoint-epoch-20" \
    --phase2_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/demo_32_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v5_EPLR_high_30epoch/checkpoints/best_model" \
    --sample_index 40 \
    --ig_steps 50 \
    --output_dir "ig_visualizations32" \
    --device "cuda"
"""


import argparse
import json
import os
import sys
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP' 
plt.rcParams['font.size'] = 10

# -----------------------------------------------------------------------------
# Import from the main script
# -----------------------------------------------------------------------------

# 必要な関数を再利用するため，元のスクリプトから関数をインポート
# 実際の使用時は，以下の関数を同じファイルにコピーするか，
# chat_gradio_integrated_gradients.pyからインポートしてください

def setup_pointllm_environment(pointllm_path: Optional[str] = None):
    """PointLLM環境をセットアップします"""
    if pointllm_path is None:
        possible_paths = [
            "PointLLM", "../PointLLM", "../../PointLLM",
            os.path.expanduser("~/PointLLM"), os.path.join(os.getcwd(), "pointllm")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pointllm_path = path
                break
    
    if pointllm_path:
        pointllm_path = os.path.abspath(pointllm_path)
        if pointllm_path not in sys.path:
            sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        from transformers import AutoConfig, AutoModelForCausalLM
        
        try:
            AutoConfig.register("pointllm", PointLLMConfig)
            AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        except Exception:
            pass
            
        print("✅ PointLLM environment setup complete")
        return True
    except ImportError as e:
        print(f"❌ Failed to setup PointLLM: {e}")
        return False


def get_device(device_name: str = "auto") -> torch.device:
    """デバイスを取得します"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_name)


def _find_matching_key(ckpt_key: str, model_state_dict: dict) -> Optional[str]:
    """チェックポイントのキーとモデルの状態辞書のキーをマッチングします"""
    if ckpt_key in model_state_dict:
        return ckpt_key
    
    clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
    candidates = [clean_key, f"model.{clean_key}", f"base_model.model.{clean_key}"]
    
    for candidate in candidates:
        if candidate in model_state_dict:
            return candidate
    
    for model_key in model_state_dict.keys():
        if model_key.endswith(clean_key) or clean_key in model_key:
            return model_key
    
    return None


def load_encoder_projector_weights(model, checkpoint_path, device, dtype):
    """Encoder + Projectorの重みをロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        if matched_key and model_state_dict[matched_key].shape == ckpt_value.shape:
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            loaded_keys.append(matched_key)
    
    if len(loaded_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
    
    print(f"   ✅ Loaded {len(loaded_keys)} parameters")


def load_projector_weights(model, checkpoint_path, device, dtype):
    """Projectorの更新された重みをロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    model_state_dict = model.state_dict()
    updated_keys = []
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        if matched_key and model_state_dict[matched_key].shape == ckpt_value.shape:
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            updated_keys.append(matched_key)
    
    if len(updated_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)


def load_llm_weights(model, checkpoint_path, device, dtype):
    """LLM部分の重みをロード"""
    print(f"      Loading LLM weights from: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    total_params = 0
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        if matched_key and model_state_dict[matched_key].shape == ckpt_value.shape:
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            loaded_keys.append(matched_key)
            total_params += ckpt_value.numel()
    
    if len(loaded_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
        print(f"   ✅ Loaded {len(loaded_keys)} LLM parameters")
        
        llm_params = sum(1 for k in loaded_keys if 'llama_model' in k or 'lm_head' in k or 'embed' in k or 'norm' in k)
        print(f"      - LLM components: {llm_params}")
        print(f"      - Total parameters: {total_params:,}")
    else:
        print(f"   ⚠️ WARNING: No LLM parameters were loaded!")


def load_full_finetuned_model(
    model_path: str,
    phase1_checkpoint_path: Optional[str] = None,
    phase2_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
):
    """2段階学習で学習したモデルの重みをロードします"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"\n{'='*60}")
    print("Loading Two-Phase Fine-tuned Model")
    print(f"{'='*60}")
    print(f"Base model: {model_path}")
    print(f"Phase 1 checkpoint: {phase1_checkpoint_path}")
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("\n[Step 1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=dtype)
    
    base_model_ref = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model_ref = model.base_model.model
    
    if hasattr(base_model_ref, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        base_model_ref.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized")
    
    # Phase 1: Encoder + Projector
    if phase1_checkpoint_path and os.path.isdir(phase1_checkpoint_path):
        print(f"\n[Phase 1] Loading Encoder + Projector weights...")
        encoder_proj_path = os.path.join(phase1_checkpoint_path, "encoder_projector.pt")
        if os.path.exists(encoder_proj_path):
            print(f"   Loading: encoder_projector.pt")
            load_encoder_projector_weights(model, encoder_proj_path, device, dtype)
        else:
            print(f"   ⚠️ Warning: encoder_projector.pt not found")
    
    # Phase 2: Projector + LLM
    if phase2_checkpoint_path and os.path.isdir(phase2_checkpoint_path):
        print(f"\n[Phase 2] Loading updated Projector + LLM weights...")
        print(f"   Path: {phase2_checkpoint_path}")
        
        # 2-1. Updated Projector
        point_proj_path = os.path.join(phase2_checkpoint_path, "point_proj.pt")
        if os.path.exists(point_proj_path):
            print(f"   Loading: point_proj.pt")
            load_projector_weights(model, point_proj_path, device, dtype)
        else:
            print(f"   ⚠️ Warning: point_proj.pt not found")
        
        # 2-2. LLM weights
        loaded_llm = False
        llm_patterns = [
            ("proj_llm_weights.pt", "projection LLM weights"),
            ("point_llm_weight.pt", "legacy format"),
            ("llm_weights.pt", "LLM weights"),
        ]
        
        for filename, description in llm_patterns:
            llm_path = os.path.join(phase2_checkpoint_path, filename)
            if os.path.exists(llm_path):
                print(f"   Loading: {filename} ({description})")
                load_llm_weights(model, llm_path, device, dtype)
                loaded_llm = True
                break
        
        if not loaded_llm:
            print(f"   ⚠️ Warning: No LLM weights found in Phase 2 checkpoint")
            print(f"   💡 Searched for: {[p[0] for p in llm_patterns]}")
    
    model.eval()
    print("\n✅ Model loaded successfully")
    return model, tokenizer


def load_and_process_point_cloud(point_cloud_path: str, num_points: int = 8192) -> torch.Tensor:
    """点群を読み込んで前処理します"""
    point_cloud = np.load(point_cloud_path)
    
    if point_cloud.shape[1] < 6:
        padding = np.ones((point_cloud.shape[0], 6 - point_cloud.shape[1])) * 0.5
        point_cloud = np.concatenate([point_cloud, padding], axis=1)
    
    point_cloud = point_cloud[:, :6]
    
    n_points = point_cloud.shape[0]
    if n_points != num_points:
        indices = np.random.choice(n_points, num_points, replace=(n_points < num_points))
        point_cloud = point_cloud[indices]
    
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
        
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    
    point_cloud = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return torch.from_numpy(point_cloud).unsqueeze(0)


def clean_question_text(content: str, point_config: dict) -> str:
    """データセット内の誤った特殊トークン形式を修正します"""
    pattern1 = r'<point_start><point_patch>\*(\d+)<point_end>'
    content = re.sub(pattern1, '', content)
    pattern2 = r'<point><point_patch>\*(\d+)'
    content = re.sub(pattern2, '', content)
    pattern3 = r'<point_patch>\*(\d+)'
    content = re.sub(pattern3, '', content)
    
    content = content.replace('<point_start>', '').replace('<point_end>', '')
    content = content.replace('<point_patch>', '').replace('<point>', '')
    return content.strip()


def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """テストデータセットJSONを読み込みます"""
    test_json_path = test_json
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, test_json)
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, "annotations", test_json)
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'data' in data:
        return data
    elif isinstance(data, list):
        return {'data': data}
    else:
        return {'data': []}


# -----------------------------------------------------------------------------
# Integrated Gradients Analyzer
# -----------------------------------------------------------------------------

class IntegratedGradientsAnalyzer:
    """Integrated Gradientsを用いた点群帰属分析クラス"""
    
    def __init__(self, model, tokenizer, device: torch.device, dtype: torch.dtype = torch.float32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.model.eval()
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
    
    def _prepare_prompt(self, question: str) -> str:
        """プロンプトを構築"""
        from pointllm.conversation import conv_templates
        
        question = clean_question_text(question, self.point_config)
        
        point_token_len = self.point_config['point_token_len']
        default_point_patch_token = self.point_config['default_point_patch_token']
        mm_use_point_start_end = self.point_config.get('mm_use_point_start_end', False)
        
        if mm_use_point_start_end:
            start = self.point_config['default_point_start_token']
            end = self.point_config['default_point_end_token']
            prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
        else:
            prompt = f"{default_point_patch_token * point_token_len}\n{question}"
        
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def _tokenize_target(self, target_text: str) -> torch.Tensor:
        """正解テキストをトークン化"""
        tokens = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
        return tokens.input_ids.to(self.device)
    
    def compute_loss(self, point_cloud: torch.Tensor, question: str, target_text: str) -> torch.Tensor:
        """モデルの損失を計算"""
        full_prompt = self._prepare_prompt(question)
        inputs = self.tokenizer([full_prompt], return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        target_ids = self._tokenize_target(target_text)
        
        labels = torch.full_like(input_ids, -100)
        labels = torch.cat([labels, target_ids], dim=1)
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        
        outputs = self.model(
            input_ids=full_input_ids,
            point_clouds=point_cloud,
            labels=labels,
            return_dict=True
        )
        return outputs.loss
    
    def compute_integrated_gradients(
        self,
        point_cloud: torch.Tensor,
        question: str,
        target_text: str,
        baseline: Optional[torch.Tensor] = None,
        baseline_type: str = "mean_color",
        steps: int = 50
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Integrated Gradientsを計算
        
        Args:
            baseline_type: "zero", "mean_color", "gray", "random"
        
        Returns:
            integrated_gradients: 統合勾配 [N, 6]
            info: ベースライン情報などのメタデータ
        """
        point_cloud = point_cloud.to(device=self.device, dtype=self.dtype)
        
        # ベースラインの設定
        if baseline is None:
            if baseline_type == "zero":
                baseline = torch.zeros_like(point_cloud)
            elif baseline_type == "mean_color":
                # 座標は保持，色を平均色に
                baseline = point_cloud.clone()
                mean_rgb = point_cloud[:, :, 3:6].mean(dim=1, keepdim=True)
                baseline[:, :, 3:6] = mean_rgb
            elif baseline_type == "gray":
                # 座標は保持，色を灰色（0.5）に
                baseline = point_cloud.clone()
                baseline[:, :, 3:6] = 0.5
            elif baseline_type == "random":
                # 各点をランダムな色に
                baseline = point_cloud.clone()
                baseline[:, :, 3:6] = torch.rand_like(point_cloud[:, :, 3:6])
            else:
                baseline = torch.zeros_like(point_cloud)
        
        baseline = baseline.to(device=self.device, dtype=self.dtype)
        
        accumulated_gradients = torch.zeros_like(point_cloud)
        
        print(f"Computing Integrated Gradients (steps={steps}, baseline={baseline_type})...")
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated = baseline + alpha * (point_cloud - baseline)
            interpolated = interpolated.clone().detach()
            interpolated.requires_grad = True
            
            loss = self.compute_loss(interpolated, question, target_text)
            loss.backward()
            
            if interpolated.grad is not None:
                accumulated_gradients += interpolated.grad.detach()
            
            self.model.zero_grad()
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{steps}")
        
        avg_gradients = accumulated_gradients / steps
        integrated_gradients = (point_cloud - baseline) * avg_gradients
        
        info = {
            'baseline_type': baseline_type,
            'steps': steps,
            'baseline_rgb_mean': baseline[0, :, 3:6].mean(dim=0).cpu().numpy()
        }
        
        return integrated_gradients.squeeze(0), info


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def visualize_ig_3d_importance(
    point_cloud: np.ndarray,
    integrated_gradients: np.ndarray,
    output_path: str,
    title: str = "Integrated Gradients: Point Importance"
):
    """
    3D空間上に各点の重要度を可視化
    
    Args:
        point_cloud: 点群データ [N, 6] (xyz + rgb)
        integrated_gradients: 統合勾配 [N, 6]
        output_path: 出力ファイルパス
        title: プロットタイトル
    """
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    
    # 各点の重要度スコア（6次元のL2ノルム）
    importance_scores = np.linalg.norm(integrated_gradients, axis=1)
    
    # 正規化（0-1範囲）
    importance_norm = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-10)
    
    fig = plt.figure(figsize=(15, 5))
    
    # サブプロット1: 元のRGB色で表示
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1, alpha=0.6)
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # サブプロット2: IG重要度でカラーマップ表示
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=importance_scores,
        cmap='hot',
        s=2,
        alpha=0.8
    )
    ax2.set_title("IG Importance (Hot colormap)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    plt.colorbar(scatter, ax=ax2, shrink=0.5, label='IG Score')
    
    # サブプロット3: 上位20%の重要な点のみ表示
    ax3 = fig.add_subplot(133, projection='3d')
    threshold = np.percentile(importance_scores, 80)
    important_mask = importance_scores >= threshold
    ax3.scatter(
        xyz[important_mask, 0],
        xyz[important_mask, 1],
        xyz[important_mask, 2],
        c=rgb[important_mask],
        s=5,
        alpha=0.9
    )
    ax3.set_title(f"Top 20% Important Points (n={important_mask.sum()})")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 3D visualization saved to: {output_path}")


def visualize_ig_6d_analysis(
    point_cloud: np.ndarray,
    integrated_gradients: np.ndarray,
    output_path: str,
    title: str = "Integrated Gradients: 6D Component Analysis"
):
    """
    6次元の各成分（x,y,z,r,g,b）の寄与を詳細分析
    
    Args:
        point_cloud: 点群データ [N, 6]
        integrated_gradients: 統合勾配 [N, 6]
        output_path: 出力ファイルパス
        title: プロットタイトル
    """
    dimension_names = ['X', 'Y', 'Z', 'R', 'G', 'B']
    
    fig = plt.figure(figsize=(18, 10))
    
    # 各次元の統計
    abs_ig = np.abs(integrated_gradients)
    mean_contrib = abs_ig.mean(axis=0)
    std_contrib = abs_ig.std(axis=0)
    
    # サブプロット1: 各次元の平均寄与度
    ax1 = plt.subplot(2, 3, 1)
    colors = ['red', 'green', 'blue', 'darkred', 'darkgreen', 'darkblue']
    bars = ax1.bar(dimension_names, mean_contrib, color=colors, alpha=0.7)
    ax1.set_ylabel('Mean |IG|')
    ax1.set_title('Average Contribution by Dimension')
    ax1.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for bar, val in zip(bars, mean_contrib):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # サブプロット2: 各次元の寄与度分布（ヒストグラム）
    ax2 = plt.subplot(2, 3, 2)
    for i, (name, color) in enumerate(zip(dimension_names, colors)):
        ax2.hist(integrated_gradients[:, i], bins=50, alpha=0.5, label=name, color=color)
    ax2.set_xlabel('IG Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('IG Distribution by Dimension')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # サブプロット3: XYZ vs RGB の寄与度比較
    ax3 = plt.subplot(2, 3, 3)
    xyz_contrib = abs_ig[:, :3].sum(axis=1).mean()
    rgb_contrib = abs_ig[:, 3:6].sum(axis=1).mean()
    ax3.bar(['Geometry (XYZ)', 'Color (RGB)'], [xyz_contrib, rgb_contrib],
            color=['steelblue', 'coral'], alpha=0.7)
    ax3.set_ylabel('Mean Total |IG|')
    ax3.set_title('Geometry vs Color Contribution')
    ax3.grid(axis='y', alpha=0.3)
    
    # 値を表示
    for i, (label, val) in enumerate(zip(['Geometry (XYZ)', 'Color (RGB)'], [xyz_contrib, rgb_contrib])):
        ax3.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # サブプロット4-6: 各次元ペアの散布図
    xyz = point_cloud[:, :3]
    
    # X-Y平面
    ax4 = plt.subplot(2, 3, 4)
    scatter4 = ax4.scatter(xyz[:, 0], xyz[:, 1], c=abs_ig[:, 0], cmap='Reds', s=1, alpha=0.6)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('X Contribution in XY Plane')
    plt.colorbar(scatter4, ax=ax4, label='|IG_x|')
    
    # Y-Z平面
    ax5 = plt.subplot(2, 3, 5)
    scatter5 = ax5.scatter(xyz[:, 1], xyz[:, 2], c=abs_ig[:, 1], cmap='Greens', s=1, alpha=0.6)
    ax5.set_xlabel('Y')
    ax5.set_ylabel('Z')
    ax5.set_title('Y Contribution in YZ Plane')
    plt.colorbar(scatter5, ax=ax5, label='|IG_y|')
    
    # X-Z平面
    ax6 = plt.subplot(2, 3, 6)
    scatter6 = ax6.scatter(xyz[:, 0], xyz[:, 2], c=abs_ig[:, 2], cmap='Blues', s=1, alpha=0.6)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    ax6.set_title('Z Contribution in XZ Plane')
    plt.colorbar(scatter6, ax=ax6, label='|IG_z|')
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 6D analysis saved to: {output_path}")
    
    # 統計情報を出力
    print(f"\n{'='*60}")
    print("6D Component Statistics:")
    print(f"{'='*60}")
    for i, name in enumerate(dimension_names):
        print(f"{name}: mean={mean_contrib[i]:.6f}, std={std_contrib[i]:.6f}")
    print(f"\nGeometry (XYZ) total: {xyz_contrib:.6f}")
    print(f"Color (RGB) total: {rgb_contrib:.6f}")
    print(f"Ratio (Color/Geometry): {rgb_contrib/xyz_contrib:.4f}")


def visualize_ig_color_contribution(
    point_cloud: np.ndarray,
    integrated_gradients: np.ndarray,
    output_path: str,
    title: str = "Integrated Gradients: RGB Color Contribution"
):
    """
    RGB色の寄与を詳細に可視化
    
    Args:
        point_cloud: 点群データ [N, 6]
        integrated_gradients: 統合勾配 [N, 6]
        output_path: 出力ファイルパス
        title: プロットタイトル
    """
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    ig_rgb = integrated_gradients[:, 3:6]
    
    fig = plt.figure(figsize=(18, 5))
    
    # R成分の寄与
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          c=np.abs(ig_rgb[:, 0]), cmap='Reds', s=2, alpha=0.7)
    ax1.set_title('Red Channel Contribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='|IG_R|')
    
    # G成分の寄与
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          c=np.abs(ig_rgb[:, 1]), cmap='Greens', s=2, alpha=0.7)
    ax2.set_title('Green Channel Contribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='|IG_G|')
    
    # B成分の寄与
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          c=np.abs(ig_rgb[:, 2]), cmap='Blues', s=2, alpha=0.7)
    ax3.set_title('Blue Channel Contribution')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, label='|IG_B|')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ RGB contribution visualization saved to: {output_path}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Integrated Gradients for a single test sample"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="annotations/test.json")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--phase1_checkpoint", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0,
                       help="Index of the sample to visualize")
    parser.add_argument("--ig_steps", type=int, default=50,
                       help="Number of integration steps")
    parser.add_argument("--baseline_type", type=str, default="mean_color",
                       choices=["zero", "mean_color", "gray", "random"],
                       help="Baseline type for IG: zero (all zeros), mean_color (preserve geometry, mean RGB), gray (preserve geometry, 0.5 RGB), random (random colors)")
    parser.add_argument("--output_dir", type=str, default="ig_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--pointllm_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 環境セットアップ
    setup_pointllm_environment(args.pointllm_path)
    device = get_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"\nUsing device: {device}")
    print(f"Using dtype: {dtype}")
    
    # モデルのロード
    model, tokenizer = load_full_finetuned_model(
        args.model_path,
        phase1_checkpoint_path=args.phase1_checkpoint,
        phase2_checkpoint_path=args.phase2_checkpoint,
        device=device,
        dtype=dtype
    )
    
    # データセットのロード
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    samples = dataset['data']
    
    if args.sample_index >= len(samples):
        raise ValueError(f"Sample index {args.sample_index} out of range (0-{len(samples)-1})")
    
    sample = samples[args.sample_index]
    sample_id = sample.get('id', f'sample_{args.sample_index}')
    
    print(f"\n{'='*60}")
    print(f"Processing Sample: {sample_id}")
    print(f"{'='*60}")
    
    # 点群のロード
    pc_rel_path = sample['point_cloud']
    pc_path = os.path.join(args.dataset_path, pc_rel_path)
    pc_tensor = load_and_process_point_cloud(pc_path, args.num_points)
    
    # 質問と正解の抽出
    conversations = sample.get('conversations', [])
    question = ""
    ground_truth = ""
    
    for turn in conversations:
        if turn['role'] == 'user':
            content = turn['content']
            lines = content.split('\n')
            if len(lines) > 1 and any(token in lines[0] for token in ['<point', 'point>']):
                question = '\n'.join(lines[1:])
            else:
                question = content
        elif turn['role'] == 'assistant':
            ground_truth = turn['content']
    
    print(f"\nQuestion: {question}")
    print(f"Ground Truth: {ground_truth}")
    
    # Integrated Gradientsの計算
    analyzer = IntegratedGradientsAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )
    
    pc_device = pc_tensor.to(device=device, dtype=dtype)
    
    integrated_gradients, ig_info = analyzer.compute_integrated_gradients(
        point_cloud=pc_device,
        question=question,
        target_text=ground_truth,
        baseline=None,
        baseline_type=args.baseline_type,
        steps=args.ig_steps
    )
    
    # numpy配列に変換
    ig_numpy = integrated_gradients.cpu().numpy()
    pc_numpy = pc_tensor.squeeze(0).cpu().numpy()
    
    print(f"\n{'='*60}")
    print("IG Computation Info:")
    print(f"{'='*60}")
    print(f"Baseline type: {ig_info['baseline_type']}")
    print(f"Integration steps: {ig_info['steps']}")
    print(f"Baseline RGB mean: {ig_info['baseline_rgb_mean']}")
    
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}")
    
    # 可視化1: 3D空間上の点の重要度
    output_3d = os.path.join(args.output_dir, f"{sample_id}_ig_3d_importance.png")
    visualize_ig_3d_importance(
        point_cloud=pc_numpy,
        integrated_gradients=ig_numpy,
        output_path=output_3d,
        title=f"IG 3D Importance: {sample_id}"
    )
    
    # 可視化2: 6次元の詳細分析
    output_6d = os.path.join(args.output_dir, f"{sample_id}_ig_6d_analysis.png")
    visualize_ig_6d_analysis(
        point_cloud=pc_numpy,
        integrated_gradients=ig_numpy,
        output_path=output_6d,
        title=f"IG 6D Analysis: {sample_id}"
    )
    
    # 可視化3: RGB色の寄与
    output_rgb = os.path.join(args.output_dir, f"{sample_id}_ig_rgb_contribution.png")
    visualize_ig_color_contribution(
        point_cloud=pc_numpy,
        integrated_gradients=ig_numpy,
        output_path=output_rgb,
        title=f"IG RGB Contribution: {sample_id}"
    )
    
    print(f"\n{'='*60}")
    print("✅ All visualizations completed successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample ID: {sample_id}")
    print(f"Integration steps: {args.ig_steps}")


if __name__ == "__main__":
    main()