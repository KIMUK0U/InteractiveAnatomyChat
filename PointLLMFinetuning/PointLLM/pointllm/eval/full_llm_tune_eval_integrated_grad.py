"""
chat_gradio_integrated_gradients.py

Integrated Gradientsを用いて点群の各要素に対する帰属度を分析します．
2段階学習済みモデルの推論と，色別のIntegrated Gradients分析を実行します．

使用方法:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python pointllm/eval/full_llm_tune_eval_integrated_grad.py \
        --dataset_path "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset" \
        --test_json "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset/annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --phase1_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_ddp_demo_ar_410_v1/checkpoints/checkpoint-epoch-20" \
        --phase2_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/demo_410_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v4_EPLR_high_30epoch/checkpoints/best_model" \
        --output_file "ig_results_410.xlsx" \
        --enable_ig_analysis \
        --ig_output_csv "ig_analysis_by_color.csv" \
        --ig_steps 50 \
        --device "cuda"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python pointllm/eval/full_llm_tune_eval_integrated_grad.py \
        --dataset_path "/home/yyamashita/Desktop/kkimu/test/demo_32_ar_instraction_dataset" \
        --test_json "/home/yyamashita/Desktop/kkimu/test/demo_32_ar_instraction_dataset/annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --phase1_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_v9_demo_dental_model_32_final/checkpoints/checkpoint-epoch-20" \
        --phase2_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/demo_32_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v5_EPLR_high_30epoch/checkpoints/best_model" \
        --output_file "ig_results_32.xlsx" \
        --enable_ig_analysis \
        --ig_output_csv "32ig_analysis_by_color.csv" \
        --ig_steps 50 \
        --device "cuda"
"""

import argparse
import json
import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union, Tuple, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------

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
        print("Please ensure PointLLM is installed or specify --pointllm_path")
        return False

def get_device(device_name: str = "auto", verbose: bool = False) -> torch.device:
    """デバイスを取得します"""
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(device_name)
        except Exception:
            device = torch.device("cpu")
    
    if verbose:
        print(f"Using device: {device}")
        
    return device

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def _find_matching_key(ckpt_key: str, model_state_dict: dict) -> Optional[str]:
    """チェックポイントのキーとモデルの状態辞書のキーをマッチングします"""
    if ckpt_key in model_state_dict:
        return ckpt_key
    
    clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
    
    candidates = [
        clean_key,
        f"model.{clean_key}",
        f"base_model.model.{clean_key}",
    ]
    
    for candidate in candidates:
        if candidate in model_state_dict:
            return candidate
    
    for model_key in model_state_dict.keys():
        if model_key.endswith(clean_key):
            return model_key
        if clean_key in model_key:
            return model_key
    
    return None

# -----------------------------------------------------------------------------
# 3. Model Loading Functions (元のコードから移植)
# -----------------------------------------------------------------------------

def load_full_finetuned_model(
    model_path: str,
    phase1_checkpoint_path: Optional[str] = None,
    phase2_checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16
):
    """2段階学習で学習したモデルの重みをロードします"""
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
    
    if phase1_checkpoint_path and os.path.isdir(phase1_checkpoint_path):
        print(f"\n[Phase 1] Loading Encoder + Projector weights...")
        encoder_proj_path = os.path.join(phase1_checkpoint_path, "encoder_projector.pt")
        if os.path.exists(encoder_proj_path):
            load_encoder_projector_weights(model, encoder_proj_path, device, dtype)
    
    if phase2_checkpoint_path and os.path.isdir(phase2_checkpoint_path):
        print(f"\n[Phase 2] Loading updated Projector + LLM weights...")
        
        point_proj_path = os.path.join(phase2_checkpoint_path, "point_proj.pt")
        if os.path.exists(point_proj_path):
            load_projector_weights(model, point_proj_path, device, dtype)
        
        loaded_llm = False
        llm_weight_patterns = [
            ("proj_llm_weights.pt", "projection LLM weights"),
            ("point_llm_weight.pt", "legacy format"),
            ("llm_weights.pt", "LLM weights"),
        ]
        
        for filename, description in llm_weight_patterns:
            llm_weight_path = os.path.join(phase2_checkpoint_path, filename)
            if os.path.exists(llm_weight_path):
                load_llm_weights(model, llm_weight_path, device, dtype)
                loaded_llm = True
                break
    
    model.eval()
    print("\n✅ Model loaded successfully")
    return model, tokenizer


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
    
    print(f"   ✅ Updated {len(updated_keys)} Projector parameters")


def load_llm_weights(model, checkpoint_path, device, dtype):
    """LLM部分の重みをロード"""
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
        print(f"   ✅ Loaded {len(loaded_keys)} LLM parameters")

# -----------------------------------------------------------------------------
# 4. Data Processing
# -----------------------------------------------------------------------------

def load_and_process_point_cloud(point_cloud_path: str, num_points: int = 8192) -> torch.Tensor:
    """点群を読み込んで前処理します"""
    try:
        point_cloud = np.load(point_cloud_path)
    except Exception as e:
        raise ValueError(f"Failed to load numpy array from {point_cloud_path}: {e}")
    
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
    
    content = content.replace('<point_start>', '')
    content = content.replace('<point_end>', '')
    content = content.replace('<point_patch>', '')
    content = content.replace('<point>', '')
    
    return content.strip()


def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """テストデータセットJSONを読み込みます"""
    test_json_path = test_json
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, test_json)
    
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, "annotations", test_json)

    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON not found: {test_json}")

    print(f"Loading dataset from: {test_json_path}")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict) and 'data' in data:
        return data
    elif isinstance(data, list):
        return {'data': data}
    else:
        return {'data': []}

# -----------------------------------------------------------------------------
# 5. Inference Logic
# -----------------------------------------------------------------------------

def generate_response(
    model, tokenizer, point_cloud, question, device, dtype,
    max_new_tokens=512, temperature=0.7, top_p=0.9
):
    """推論を実行して応答を生成します"""
    if hasattr(model, "get_model"):
        point_config = model.get_model().point_backbone_config
    elif hasattr(model, "base_model") and hasattr(model.base_model, "get_model"):
        point_config = model.base_model.get_model().point_backbone_config
    else:
        point_config = getattr(model.config, "point_backbone_config", None)
        if point_config is None:
            point_config = model.base_model.model.point_backbone_config

    point_token_len = point_config['point_token_len']
    default_point_patch_token = point_config['default_point_patch_token']
    mm_use_point_start_end = point_config.get('mm_use_point_start_end', False)
    
    question = clean_question_text(question, point_config)
    
    if mm_use_point_start_end:
        start = point_config['default_point_start_token']
        end = point_config['default_point_end_token']
        prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
    else:
        prompt = f"{default_point_patch_token * point_token_len}\n{question}"
    
    from pointllm.conversation import conv_templates, SeparatorStyle
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    inputs = tokenizer([full_prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    point_cloud = point_cloud.to(device=device, dtype=dtype)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_cloud,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        
    input_len = input_ids.shape[1]
    response = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if response.endswith(stop_str):
        response = response[:-len(stop_str)]
        
    return response.strip()

# -----------------------------------------------------------------------------
# 6. Integrated Gradients Implementation
# -----------------------------------------------------------------------------

class IntegratedGradientsAnalyzer:
    """Integrated Gradientsを用いた点群帰属分析クラス"""
    
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
        target_text: str
    ) -> torch.Tensor:
        """
        モデルの損失を計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6]
            question: 質問文
            target_text: 正解テキスト
            
        Returns:
            loss: 損失値
        """
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
        steps: int = 50
    ) -> torch.Tensor:
        """
        Integrated Gradientsを計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6]
            question: 質問文
            target_text: 正解テキスト
            baseline: ベースライン [1, N, 6]（Noneの場合は零ベクトル）
            steps: 積分ステップ数
            
        Returns:
            integrated_gradients: 統合勾配 [N, 6]
        """
        # ベースラインの設定（デフォルトは零ベクトル）
        if baseline is None:
            baseline = torch.zeros_like(point_cloud)
        
        baseline = baseline.to(device=self.device, dtype=self.dtype)
        point_cloud = point_cloud.to(device=self.device, dtype=self.dtype)
        
        # 各ステップでの勾配を累積
        accumulated_gradients = torch.zeros_like(point_cloud)
        
        # 積分パスを生成（ベースラインから入力まで）
        for step in range(steps):
            # 補間係数 alpha
            alpha = (step + 1) / steps
            
            # 補間された点群
            interpolated = baseline + alpha * (point_cloud - baseline)
            interpolated = interpolated.clone().detach()
            interpolated.requires_grad = True
            
            # 損失計算
            loss = self.compute_loss(interpolated, question, target_text)
            
            # 勾配計算
            loss.backward()
            
            # 勾配を累積
            if interpolated.grad is not None:
                accumulated_gradients += interpolated.grad.detach()
            
            # 勾配をクリア
            self.model.zero_grad()
        
        # 平均勾配を計算
        avg_gradients = accumulated_gradients / steps
        
        # Integrated Gradientsの計算: (x - baseline) * avg_gradient
        integrated_gradients = (point_cloud - baseline) * avg_gradients
        
        # [1, N, 6] -> [N, 6]
        integrated_gradients = integrated_gradients.squeeze(0)
        
        return integrated_gradients
    
    def compute_importance_scores(
        self,
        integrated_gradients: torch.Tensor,
        aggregation: Literal["l2_norm", "abs_sum", "max_abs", "mean_abs"] = "l2_norm"
    ) -> np.ndarray:
        """
        統合勾配から各点の重要度スコアを計算
        
        Args:
            integrated_gradients: 統合勾配テンソル [N, 6]
            aggregation: 集約方法
                
        Returns:
            importance_scores: 各点の重要度 [N]
        """
        if aggregation == "l2_norm":
            scores = torch.norm(integrated_gradients, p=2, dim=1)
        elif aggregation == "abs_sum":
            scores = torch.sum(torch.abs(integrated_gradients), dim=1)
        elif aggregation == "max_abs":
            scores = torch.max(torch.abs(integrated_gradients), dim=1)[0]
        elif aggregation == "mean_abs":
            scores = torch.mean(torch.abs(integrated_gradients), dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return scores.cpu().numpy()


# -----------------------------------------------------------------------------
# 7. Color Detection and Segmentation
# -----------------------------------------------------------------------------

def detect_target_colors_in_question(question: str) -> List[str]:
    """質問文から対象となる色を検出"""
    question_lower = question.lower()
    
    color_patterns = {
        'red': ['red', '赤'],
        'blue': ['blue', '青'],
        'dark_gray': ['dark gray', 'dark grey', 'darkgray', 'darkgrey', 'black', '暗灰色']
    }
    
    detected = []
    for color_name, patterns in color_patterns.items():
        if any(pattern in question_lower for pattern in patterns):
            detected.append(color_name)
    
    return detected


def get_color_reference_rgb(color_name: str) -> np.ndarray:
    """色名から参照RGB値を取得"""
    color_map = {
        'red': np.array([1.0, 0.0, 0.0]),
        'blue': np.array([0.0, 0.0, 1.0]),
        'dark_gray': np.array([0.0, 0.0, 0.0])
    }
    
    return color_map.get(color_name, np.array([0.5, 0.5, 0.5]))


def compute_color_membership(
    point_cloud: np.ndarray,
    target_color_rgb: np.ndarray,
    sigma: float = 0.2
) -> np.ndarray:
    """ガウス減衰関数による色の所属度を計算"""
    rgb = point_cloud[:, 3:6]
    distances = np.linalg.norm(rgb - target_color_rgb, axis=1)
    membership = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    return membership


def segment_points_by_color(
    point_cloud: np.ndarray,
    detected_colors: List[str],
    threshold: float = 0.5,
    sigma: float = 0.2
) -> Dict[str, np.ndarray]:
    """色ごとに点群をセグメンテーション"""
    segmentation = {}
    
    for color_name in detected_colors:
        target_rgb = get_color_reference_rgb(color_name)
        membership = compute_color_membership(point_cloud, target_rgb, sigma)
        mask = membership >= threshold
        segmentation[color_name] = mask
    
    all_color_masks = np.zeros(len(point_cloud), dtype=bool)
    for mask in segmentation.values():
        all_color_masks |= mask
    
    segmentation['other'] = ~all_color_masks
    
    return segmentation


# -----------------------------------------------------------------------------
# 8. Integrated Gradients Analysis with Color Segmentation
# -----------------------------------------------------------------------------

def analyze_integrated_gradients_by_color(
    model,
    tokenizer,
    point_cloud: torch.Tensor,
    question: str,
    target_text: str,
    device: torch.device,
    dtype: torch.dtype,
    ig_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    aggregation: str = "l2_norm"
) -> Dict[str, Any]:
    """
    色別にIntegrated Gradientsを分析
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群テンソル [1, 8192, 6]
        question: 質問文
        target_text: 正解テキスト
        device: デバイス
        dtype: データ型
        ig_steps: Integrated Gradientsの積分ステップ数
        baseline: ベースライン（Noneの場合は零ベクトル）
        aggregation: スコア集約方法
        
    Returns:
        analysis_result: 分析結果の辞書
    """
    detected_colors = detect_target_colors_in_question(question)
    
    if not detected_colors:
        return None
    
    analyzer = IntegratedGradientsAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )
    
    point_cloud_device = point_cloud.to(device=device, dtype=dtype)
    
    # Integrated Gradientsを計算
    integrated_gradients = analyzer.compute_integrated_gradients(
        point_cloud=point_cloud_device,
        question=question,
        target_text=target_text,
        baseline=baseline,
        steps=ig_steps
    )
    
    # 重要度スコアを計算
    importance_scores = analyzer.compute_importance_scores(
        integrated_gradients, 
        aggregation=aggregation
    )
    
    pc_numpy = point_cloud.squeeze(0).cpu().numpy()
    
    # 色別セグメンテーション
    segmentation = segment_points_by_color(pc_numpy, detected_colors)
    
    # 各色カテゴリの平均IG値を計算
    color_ig_means = {}
    color_point_counts = {}
    
    for color_name, mask in segmentation.items():
        if mask.sum() > 0:
            color_ig_means[color_name] = importance_scores[mask].mean()
            color_point_counts[color_name] = mask.sum()
        else:
            color_ig_means[color_name] = 0.0
            color_point_counts[color_name] = 0
    
    return {
        'detected_colors': detected_colors,
        'color_ig_means': color_ig_means,
        'color_point_counts': color_point_counts,
        'total_points': len(pc_numpy)
    }


# -----------------------------------------------------------------------------
# 9. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Integrated Gradients Analysis"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="annotations/test.json")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--phase1_checkpoint", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="ig_results.xlsx")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--pointllm_path", type=str, default=None)
    
    parser.add_argument("--enable_ig_analysis", action="store_true")
    parser.add_argument("--ig_output_csv", type=str, default="ig_analysis_by_color.csv")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--ig_aggregation", type=str, default="l2_norm",
                       choices=["l2_norm", "abs_sum", "max_abs", "mean_abs"])
    
    args = parser.parse_args()
    
    setup_pointllm_environment(args.pointllm_path)
    device = get_device(args.device, verbose=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using dtype: {dtype}")
    
    model, tokenizer = load_full_finetuned_model(
        args.model_path, 
        phase1_checkpoint_path=args.phase1_checkpoint,
        phase2_checkpoint_path=args.phase2_checkpoint,
        device=device, 
        dtype=dtype
    )
    
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    samples = dataset['data']
    print(f"\nFound {len(samples)} samples for inference")
    
    results = []
    ig_results = []
    color_ig_accumulator = defaultdict(list)
    
    print(f"\n{'='*60}")
    print("Starting Batch Inference with Integrated Gradients")
    if args.enable_ig_analysis:
        print(f"IG Analysis: ENABLED (steps={args.ig_steps})")
        print(f"IG Output: {args.ig_output_csv}")
    print(f"{'='*60}\n")
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        sample_id = sample.get('id', f'sample_{i}')
        pc_rel_path = sample['point_cloud']
        pc_path = os.path.join(args.dataset_path, pc_rel_path)
        
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
        
        if not question:
            question = "Describe this 3D object."
            
        try:
            pc_tensor = load_and_process_point_cloud(pc_path, args.num_points)
            
            response = generate_response(
                model, tokenizer, pc_tensor, question, device, dtype,
                max_new_tokens=args.max_new_tokens
            )
            
            if args.enable_ig_analysis and ground_truth:
                try:
                    analysis = analyze_integrated_gradients_by_color(
                        model=model,
                        tokenizer=tokenizer,
                        point_cloud=pc_tensor,
                        question=question,
                        target_text=ground_truth,
                        device=device,
                        dtype=dtype,
                        ig_steps=args.ig_steps,
                        aggregation=args.ig_aggregation
                    )
                    
                    if analysis is not None:
                        ig_row = {
                            'sample_id': sample_id,
                            'detected_colors': ','.join(analysis['detected_colors']),
                            'total_points': analysis['total_points']
                        }
                        
                        for color_name in ['red', 'blue', 'dark_gray', 'other']:
                            ig_mean = analysis['color_ig_means'].get(color_name, 0.0)
                            point_count = analysis['color_point_counts'].get(color_name, 0)
                            
                            ig_row[f'{color_name}_ig_mean'] = ig_mean
                            ig_row[f'{color_name}_point_count'] = point_count
                            
                            if point_count > 0:
                                color_ig_accumulator[color_name].append(ig_mean)
                        
                        ig_results.append(ig_row)
                        
                except Exception as e:
                    print(f"\n⚠️ IG analysis failed for {sample_id}: {e}")
                    
        except Exception as e:
            print(f"\n❌ Error processing {sample_id}: {e}")
            response = f"ERROR: {str(e)}"
            
        results.append({
            "id": sample_id,
            "point_cloud": pc_rel_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response
        })
        
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_excel(args.output_file, index=False)
            if args.enable_ig_analysis and ig_results:
                pd.DataFrame(ig_results).to_csv(args.ig_output_csv, index=False)
            print(f"\n💾 Intermediate save: {i+1}/{len(samples)} samples processed")
            
    df = pd.DataFrame(results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Inference completed successfully!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_file}")
    print(f"Total samples: {len(results)}")
    
    if args.enable_ig_analysis and ig_results:
        ig_df = pd.DataFrame(ig_results)
        ig_df.to_csv(args.ig_output_csv, index=False)
        
        overall_stats = {}
        for color_name in ['red', 'blue', 'dark_gray', 'other']:
            if color_name in color_ig_accumulator and color_ig_accumulator[color_name]:
                overall_stats[f'{color_name}_mean_ig'] = np.mean(color_ig_accumulator[color_name])
                overall_stats[f'{color_name}_std_ig'] = np.std(color_ig_accumulator[color_name])
                overall_stats[f'{color_name}_sample_count'] = len(color_ig_accumulator[color_name])
            else:
                overall_stats[f'{color_name}_mean_ig'] = 0.0
                overall_stats[f'{color_name}_std_ig'] = 0.0
                overall_stats[f'{color_name}_sample_count'] = 0
        
        stats_df = pd.DataFrame([overall_stats])
        stats_df.insert(0, 'sample_id', 'OVERALL_STATISTICS')
        
        combined_df = pd.concat([ig_df, stats_df], ignore_index=True)
        combined_df.to_csv(args.ig_output_csv, index=False)
        
        print(f"\n{'='*60}")
        print("Integrated Gradients Analysis Summary")
        print(f"{'='*60}")
        print(f"IG analysis saved to: {args.ig_output_csv}")
        print(f"Analyzed samples: {len(ig_results)}")
        print(f"Integration steps: {args.ig_steps}")
        print(f"\nOverall Statistics:")
        for color_name in ['red', 'blue', 'dark_gray', 'other']:
            mean_ig = overall_stats.get(f'{color_name}_mean_ig', 0.0)
            std_ig = overall_stats.get(f'{color_name}_std_ig', 0.0)
            count = overall_stats.get(f'{color_name}_sample_count', 0)
            print(f"  {color_name:12s}: mean={mean_ig:.6f}, std={std_ig:.6f}, n={count}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()