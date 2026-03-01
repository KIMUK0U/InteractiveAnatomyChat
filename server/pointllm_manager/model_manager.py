"""
PointLLMモデルの管理クラス（Encoder + Proj + LoRA + LLM Finetune対応版）
起動時に一度だけモデルをロードし、推論時に再利用

対応する学習パターン:
1. 2段階学習（Encoder+Proj → Proj+LoRA）
   - encoder_checkpoint_path: Encoder + Projectorの学習済み重み
   - lora_checkpoint_path: Projector更新分 + LoRAアダプタの重み
   
2. 1段階学習（Proj+LoRAのみ）
   - encoder_checkpoint_path: None
   - lora_checkpoint_path: Projector + LoRAアダプタの重み（Encoder凍結）

3. 2段階学習（Encoder+Proj → Proj+LLM）★新規追加
   - encoder_checkpoint_path: Encoder + Projectorの学習済み重み
   - llm_checkpoint_path: Projector更新分 + LLMファインチューニング済み重み

重みのロード順序:
1. ベースモデル（PointLLM）をロード
2. [オプション] Stage 1の重み（Encoder + Projector）をロード
3. LoRAアダプタをロード OR LLMファインチューニング済み重みをロード
4. Projectorの更新分をロード
"""

import torch
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

# PointLLMのパス設定
POINTLLM_ROOT = Path(__file__).parent.parent / "PointLLM"
sys.path.insert(0, str(POINTLLM_ROOT))

from pointllm.eval.chat_gradio_finetuning_encoder_proj_LoRA import (
    setup_pointllm_environment,
    get_device
)


class ModelManager:
    """PointLLMモデル管理クラス（Encoder + Proj + LoRA + LLM Finetune対応）"""
    
    def __init__(
        self,
        model_path: str = "RunsenXu/PointLLM_7B_v1.2",
        encoder_checkpoint_path: Optional[str] = None,
        lora_checkpoint_path: Optional[str] = None,
        llm_checkpoint_path: Optional[str] = None,
        auto_detect_mode: bool = True
    ):
        """
        初期化
        
        Args:
            model_path: ベースモデルのパス
            encoder_checkpoint_path: Stage 1のチェックポイントディレクトリ（オプション）
                - encoder_projector.pt (Encoder + Projector)
            lora_checkpoint_path: Stage 2 LoRAのチェックポイントディレクトリ
                - point_proj.pt (Projector重み)
                - adapter_model.safetensors (LoRA)
                - adapter_config.json (LoRA設定)
            llm_checkpoint_path: Stage 2 LLM Finetuneのチェックポイントディレクトリ★新規追加
                - point_proj.pt (Projector重み)
                - model-*.safetensors (LLMファインチューニング済み重み)
                - config.json
            auto_detect_mode: チェックポイントから学習モードを自動検出
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.is_initialized = False
        
        # モデル設定
        self.model_path = model_path
        self.encoder_checkpoint_path = encoder_checkpoint_path
        self.lora_checkpoint_path = lora_checkpoint_path
        self.llm_checkpoint_path = llm_checkpoint_path
        self.auto_detect_mode = auto_detect_mode
        
        # 学習モードの判定
        self.training_mode = None  # "two_stage_lora", "proj_lora_only", "two_stage_llm"
        
        # チェックポイントの検証
        self.encoder_files = {}
        self.lora_files = {}
        self.llm_files = {}
        
        if self.encoder_checkpoint_path:
            self.encoder_files = self._validate_encoder_checkpoint()
        
        if self.lora_checkpoint_path:
            self.lora_files = self._validate_lora_checkpoint()
        
        if self.llm_checkpoint_path:
            self.llm_files = self._validate_llm_checkpoint()
        
        # 学習モードの自動検出
        if self.auto_detect_mode:
            self._detect_training_mode()
    
    def _detect_training_mode(self):
        """
        チェックポイントから学習モードを自動検出
        
        判定基準:
        - encoder_checkpoint_pathが指定され、encoder_projector.ptが存在
          かつ llm_checkpoint_pathが指定されている → "two_stage_llm"
        - encoder_checkpoint_pathが指定され、encoder_projector.ptが存在
          かつ lora_checkpoint_pathが指定されている → "two_stage_lora"
        - encoder_checkpoint_pathがNoneまたはファイルなし → "proj_lora_only"
        """
        has_encoder = self.encoder_files and 'encoder_projector' in self.encoder_files
        has_lora = self.lora_files and 'adapter_config' in self.lora_files
        has_llm = self.llm_files and 'config' in self.llm_files
        
        if has_encoder and has_llm:
            self.training_mode = "two_stage_llm"
            print(f"\n[Mode Detection] Training mode: Two-Stage (Encoder+Proj → Proj+LLM Finetune)")
        elif has_encoder and has_lora:
            self.training_mode = "two_stage_lora"
            print(f"\n[Mode Detection] Training mode: Two-Stage (Encoder+Proj → Proj+LoRA)")
        elif has_encoder:  # ★ここを追加: Encoderのみの場合に対応
            self.training_mode = "encoder_proj_only"
            print(f"\n[Mode Detection] Training mode: Encoder+Proj Only (Stage 1 only)")
        else:
            self.training_mode = "proj_lora_only"
            print(f"\n[Mode Detection] Training mode: Proj+LoRA Only (Encoder frozen)")
    
    def _validate_encoder_checkpoint(self) -> Dict[str, Path]:
        """
        Stage 1チェックポイントの検証
        
        Returns:
            Dict[str, Path]: 検証済みファイルパス
        """
        checkpoint_dir = Path(self.encoder_checkpoint_path)
        files = {}
        
        print(f"\n[Stage 1] Validating Encoder checkpoint: {checkpoint_dir}")
        
        if not checkpoint_dir.exists():
            print(f"   ⚠️ Directory not found: {checkpoint_dir}")
            return files
        
        # encoder_projector.ptの確認
        encoder_proj_path = checkpoint_dir / "encoder_projector.pt"
        if encoder_proj_path.exists():
            files['encoder_projector'] = encoder_proj_path
            print(f"   ✅ Found: encoder_projector.pt")
        else:
            print(f"   ⚠️ Missing: encoder_projector.pt")
        
        return files
    
    def _validate_lora_checkpoint(self) -> Dict[str, Path]:
        """
        Stage 2 LoRAチェックポイントの検証
        
        Returns:
            Dict[str, Path]: 検証済みファイルパス
        """
        checkpoint_dir = Path(self.lora_checkpoint_path)
        files = {}
        
        print(f"\n[Stage 2 LoRA] Validating LoRA checkpoint: {checkpoint_dir}")
        
        if not checkpoint_dir.exists():
            print(f"   ⚠️ Directory not found: {checkpoint_dir}")
            return files
        
        # point_proj.ptの確認
        point_proj_path = checkpoint_dir / "point_proj.pt"
        if point_proj_path.exists():
            files['point_proj'] = point_proj_path
            print(f"   ✅ Found: point_proj.pt")
        else:
            print(f"   ⚠️ Missing: point_proj.pt")
        
        # LoRA adapter_config.jsonの確認
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if adapter_config_path.exists():
            files['adapter_config'] = adapter_config_path
            print(f"   ✅ Found: adapter_config.json")
        else:
            print(f"   ⚠️ Missing: adapter_config.json")
        
        # LoRA adapter_model.binまたは.safetensorsの確認
        adapter_model_bin = checkpoint_dir / "adapter_model.bin"
        adapter_model_safetensors = checkpoint_dir / "adapter_model.safetensors"
        
        if adapter_model_bin.exists():
            files['adapter_model'] = adapter_model_bin
            print(f"   ✅ Found: adapter_model.bin")
        elif adapter_model_safetensors.exists():
            files['adapter_model'] = adapter_model_safetensors
            print(f"   ✅ Found: adapter_model.safetensors")
        else:
            print(f"   ⚠️ Missing: adapter_model files")
        
        return files
    
    def _validate_llm_checkpoint(self) -> Dict[str, Path]:
        """
        Stage 2 LLM Finetuneチェックポイントの検証★新規追加
        
        Returns:
            Dict[str, Path]: 検証済みファイルパス
        """
        checkpoint_dir = Path(self.llm_checkpoint_path)
        files = {}
        
        print(f"\n[Stage 2 LLM] Validating LLM Finetune checkpoint: {checkpoint_dir}")
        
        if not checkpoint_dir.exists():
            print(f"   ⚠️ Directory not found: {checkpoint_dir}")
            return files
        
        # point_proj.ptの確認
        point_proj_path = checkpoint_dir / "point_proj.pt"
        if point_proj_path.exists():
            files['point_proj'] = point_proj_path
            print(f"   ✅ Found: point_proj.pt")
        else:
            print(f"   ⚠️ Missing: point_proj.pt")
        
        # config.jsonの確認
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            files['config'] = config_path
            print(f"   ✅ Found: config.json")
        else:
            print(f"   ⚠️ Missing: config.json")
        
        # model-*.safetensorsファイルの確認
        safetensor_files = list(checkpoint_dir.glob("model-*.safetensors"))
        if safetensor_files:
            files['model_safetensors'] = safetensor_files
            print(f"   ✅ Found: {len(safetensor_files)} safetensors file(s)")
        else:
            print(f"   ⚠️ Missing: model-*.safetensors files")
        
        # model.safetensors.index.jsonの確認
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            files['index'] = index_path
            print(f"   ✅ Found: model.safetensors.index.json")
        else:
            print(f"   ⚠️ Missing: model.safetensors.index.json")
        
        return files
    
    def _load_base_model(self):
        """ベースモデルとトークナイザーをロード"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"\n[Base Model] Loading from: {self.model_path}")
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            padding_side="right",
        )
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # モデルのロード
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        
        self.model = self.model.to(self.device)
        
        # point_backbone_configの初期化
        if hasattr(self.model, 'initialize_tokenizer_point_backbone_config_wo_embedding'):
            self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
            print("   ✅ Point backbone config initialized")
        
        print(f"   ✅ Base model loaded: {type(self.model).__name__}")
    
    def _load_encoder_projector_weights(self):
        """Stage 1のEncoder + Projector重みをロード"""
        if 'encoder_projector' not in self.encoder_files:
            print("\n[Stage 1] Skipping: No encoder_projector.pt found")
            return
        
        print(f"\n[Stage 1] Loading Encoder + Projector weights...")
        
        encoder_proj_path = self.encoder_files['encoder_projector']
        checkpoint = torch.load(encoder_proj_path, map_location="cpu")
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        
        # モデルのstate_dictを取得
        model_state_dict = self.model.state_dict()
        
        # キーマッチングとロード
        loaded_keys = []
        encoder_param_count = 0
        projector_param_count = 0
        
        for ckpt_key, ckpt_value in state_dict.items():
            # "model." プレフィックスを除去して試す
            clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
            
            # マッチング候補
            candidates = [ckpt_key, clean_key, f"model.{clean_key}"]
            
            matched = False
            for candidate in candidates:
                if candidate in model_state_dict:
                    model_state_dict[candidate] = ckpt_value.to(device=self.device, dtype=self.dtype)
                    loaded_keys.append(candidate)
                    matched = True
                    
                    param_count = ckpt_value.numel()
                    if 'point_backbone' in candidate or 'encoder' in candidate:
                        encoder_param_count += param_count
                    elif 'point_proj' in candidate or 'projector' in candidate:
                        projector_param_count += param_count
                    
                    break
            
            if not matched:
                # 部分一致を試す
                for model_key in model_state_dict.keys():
                    if model_key.endswith(clean_key) or clean_key in model_key:
                        model_state_dict[model_key] = ckpt_value.to(device=self.device, dtype=self.dtype)
                        loaded_keys.append(model_key)
                        
                        param_count = ckpt_value.numel()
                        if 'point_backbone' in model_key or 'encoder' in model_key:
                            encoder_param_count += param_count
                        elif 'point_proj' in model_key or 'projector' in model_key:
                            projector_param_count += param_count
                        
                        break
        
        # 更新された状態をモデルにロード
        self.model.load_state_dict(model_state_dict, strict=False)
        
        print(f"   ✅ Loaded from: {encoder_proj_path.name}")
        print(f"   - Loaded keys: {len(loaded_keys)}")
        print(f"   - Encoder parameters: {encoder_param_count:,} ({len([k for k in loaded_keys if 'backbone' in k])} tensors)")
        print(f"   - Projector parameters: {projector_param_count:,} ({len([k for k in loaded_keys if 'proj' in k])} tensors)")
    
    def _load_projector_update(self, checkpoint_path: Path):
        """
        Projector重みをロード（プレフィックス除去対応版）
        
        Args:
            checkpoint_path: point_proj.ptのパス
        """
        print(f"\n[Projector] Loading Projector weights from: {checkpoint_path.name}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        
        # 現在のモデルの状態を取得
        model_state_dict = self.model.state_dict()
        
        updated_keys = []
        updated_param_count = 0
        
        # 不要なプレフィックスの候補リスト
        prefixes_to_strip = [
            "base_model.model.model.",
            "base_model.model.",
            "model.",
        ]

        for ckpt_key, ckpt_value in state_dict.items():
            matched_key_in_model = None
            
            # 1. 完全一致チェック
            if ckpt_key in model_state_dict:
                matched_key_in_model = ckpt_key
            
            # 2. プレフィックス除去チェック
            if not matched_key_in_model:
                for prefix in prefixes_to_strip:
                    if ckpt_key.startswith(prefix):
                        stripped_key = ckpt_key[len(prefix):]
                        
                        # "model." を補って試すケース
                        candidate_with_model = f"model.{stripped_key}"
                        
                        if stripped_key in model_state_dict:
                            matched_key_in_model = stripped_key
                            break
                        elif candidate_with_model in model_state_dict:
                            matched_key_in_model = candidate_with_model
                            break
            
            # 3. 部分一致検索（最終手段: point_projが含まれる場合）
            if not matched_key_in_model and "point_proj" in ckpt_key:
                suffix = ckpt_key.split("point_proj")[-1]
                target_suffix = f"point_proj{suffix}"
                
                for model_key in model_state_dict.keys():
                    if model_key.endswith(target_suffix):
                        matched_key_in_model = model_key
                        break

            # ロード実行
            if matched_key_in_model:
                # 形状チェック
                if model_state_dict[matched_key_in_model].shape != ckpt_value.shape:
                    print(f"   ⚠️ Shape mismatch for {matched_key_in_model}: Model {model_state_dict[matched_key_in_model].shape} vs Ckpt {ckpt_value.shape}")
                    continue
                
                model_state_dict[matched_key_in_model] = ckpt_value.to(device=self.device, dtype=self.dtype)
                updated_keys.append(matched_key_in_model)
                updated_param_count += ckpt_value.numel()
            else:
                # デバッグ用: マッチしなかったキーを最初だけ表示
                if len(updated_keys) == 0: 
                    print(f"   ⚠️ Could not match key: {ckpt_key}")

        # 更新された状態をモデルにロード
        if len(updated_keys) > 0:
            self.model.load_state_dict(model_state_dict, strict=False)
            print(f"   ✅ Loaded from: {checkpoint_path.name}")
            print(f"   - Updated keys: {len(updated_keys)}")
            print(f"   - Updated parameters: {updated_param_count:,}")
        else:
            print(f"\n   ⚠️ WARNING: No Projector parameters were loaded!")
    
    def _load_lora_adapter(self):
        """LoRAアダプタをロード"""
        if 'adapter_config' not in self.lora_files or 'adapter_model' not in self.lora_files:
            print("\n[LoRA] Skipping: Adapter files not found")
            return
        
        print(f"\n[LoRA] Loading LoRA adapter...")
        
        try:
            from peft import PeftModel
            
            # LoRAアダプタをロード
            checkpoint_dir = Path(self.lora_checkpoint_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                str(checkpoint_dir),
                is_trainable=False  # 推論モード
            )
            
            print(f"   ✅ LoRA adapter loaded from: {checkpoint_dir.name}")
            print(f"   - Model type: {type(self.model).__name__}")
            
        except ImportError:
            print("   ⚠️ peft library not installed, skipping LoRA")
        except Exception as e:
            print(f"   ⚠️ Failed to load LoRA adapter: {e}")
    
    def _validate_llm_checkpoint(self) -> Dict[str, Path]:
        """
        Stage 2 LLM Finetuneチェックポイントの検証（両形式対応）
        
        対応形式:
        1. trainer_LoRA.py形式: model.safetensors（単一ファイル）
        2. trainer_proj_llm_ddp.py形式: model-*.safetensors（分割ファイル）
        
        Returns:
            Dict[str, Path]: 検証済みファイルパス
        """
        checkpoint_dir = Path(self.llm_checkpoint_path)
        files = {}
        
        print(f"\n[Stage 2 LLM] Validating LLM Finetune checkpoint: {checkpoint_dir}")
        
        if not checkpoint_dir.exists():
            print(f"   ⚠️ Directory not found: {checkpoint_dir}")
            return files
        
        # point_proj.ptの確認
        point_proj_path = checkpoint_dir / "point_proj.pt"
        if point_proj_path.exists():
            files['point_proj'] = point_proj_path
            print(f"   ✅ Found: point_proj.pt")
        else:
            print(f"   ⚠️ Missing: point_proj.pt")
        
        # config.jsonの確認
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            files['config'] = config_path
            print(f"   ✅ Found: config.json")
        else:
            print(f"   ⚠️ Missing: config.json")
        
        # ✅ 形式1: 単一のmodel.safetensors（trainer_LoRA.py形式）
        single_safetensor = checkpoint_dir / "model.safetensors"
        if single_safetensor.exists():
            files['model_safetensors'] = [single_safetensor]
            files['format'] = 'single'
            file_size = single_safetensor.stat().st_size / 1024**3
            print(f"   ✅ Found: model.safetensors ({file_size:.2f} GB) [trainer_LoRA.py format]")
            return files
        
        # ✅ 形式2: 分割されたmodel-*.safetensors（DDP形式）
        safetensor_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
        if safetensor_files:
            files['model_safetensors'] = safetensor_files
            files['format'] = 'sharded'
            total_size = sum(f.stat().st_size for f in safetensor_files) / 1024**3
            print(f"   ✅ Found: {len(safetensor_files)} safetensors file(s) ({total_size:.2f} GB) [DDP format]")
            
            # model.safetensors.index.jsonの確認（分割形式の場合）
            index_path = checkpoint_dir / "model.safetensors.index.json"
            if index_path.exists():
                files['index'] = index_path
                print(f"   ✅ Found: model.safetensors.index.json")
            else:
                print(f"   ⚠️ Missing: model.safetensors.index.json (optional for sharded format)")
        else:
            print(f"   ⚠️ Missing: model safetensors files")
        
        # ✅ 形式3: proj_llm_weights.pt（古いDDP形式、後方互換性）
        proj_llm_weights = checkpoint_dir / "proj_llm_weights.pt"
        if proj_llm_weights.exists():
            files['proj_llm_weights'] = proj_llm_weights
            files['format'] = 'legacy_pt'
            file_size = proj_llm_weights.stat().st_size / 1024**3
            print(f"   ✅ Found: proj_llm_weights.pt ({file_size:.2f} GB) [Legacy DDP format]")
        
        return files

    def _load_llm_finetuned_weights(self):
        """
        LLMファインチューニング済み重みをロード（両形式対応）
        
        対応形式:
        1. model.safetensors（単一ファイル）
        2. model-*.safetensors（分割ファイル）
        3. proj_llm_weights.pt（レガシー形式）
        """
        if 'config' not in self.llm_files:
            print("\n[LLM Finetune] Skipping: config.json not found")
            return
        
        print(f"\n[LLM Finetune] Loading LLM finetuned weights...")
        
        checkpoint_dir = Path(self.llm_checkpoint_path)
        format_type = self.llm_files.get('format', 'unknown')
        
        print(f"   Detected format: {format_type}")
        
        try:
            # ✅ 形式1 & 2: safetensors形式（単一 or 分割）
            if 'model_safetensors' in self.llm_files:
                self._load_safetensors_weights(self.llm_files['model_safetensors'])
            
            # ✅ 形式3: レガシーのproj_llm_weights.pt
            elif 'proj_llm_weights' in self.llm_files:
                self._load_legacy_pt_weights(self.llm_files['proj_llm_weights'])
            
            else:
                print(f"   ⚠️ No recognized weight files found")
                
        except Exception as e:
            print(f"   ⚠️ Failed to load LLM finetuned weights: {e}")
            import traceback
            traceback.print_exc()

    def _load_safetensors_weights(self, safetensor_files: list[Path]):
        """
        safetensors形式の重みをロード（単一 & 分割対応）
        
        Args:
            safetensor_files: safetensorsファイルのリスト
        """
        try:
            from safetensors import safe_open
        except ImportError:
            print("   ⚠️ safetensors library not installed")
            return
        
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        total_params = 0
        
        for safetensor_file in safetensor_files:
            print(f"   Loading from: {safetensor_file.name}")
            
            with safe_open(safetensor_file, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    # キー名のマッチング（プレフィックス除去対応）
                    matched_key = self._find_matching_key(key, model_state_dict)
                    
                    if matched_key:
                        tensor = f.get_tensor(key)
                        
                        # 形状チェック
                        if model_state_dict[matched_key].shape != tensor.shape:
                            print(f"   ⚠️ Shape mismatch for {matched_key}")
                            continue
                        
                        model_state_dict[matched_key] = tensor.to(device=self.device, dtype=self.dtype)
                        loaded_keys.append(matched_key)
                        total_params += tensor.numel()
        
        if len(loaded_keys) > 0:
            self.model.load_state_dict(model_state_dict, strict=False)
            print(f"   ✅ Loaded safetensors weights")
            print(f"   - Loaded keys: {len(loaded_keys)}")
            print(f"   - Total parameters: {total_params:,}")
        else:
            print(f"   ⚠️ WARNING: No parameters were loaded from safetensors!")

    def _load_legacy_pt_weights(self, pt_file: Path):
        """
        レガシーのproj_llm_weights.ptをロード
        
        Args:
            pt_file: proj_llm_weights.ptのパス
        """
        print(f"   Loading legacy format from: {pt_file.name}")
        
        checkpoint = torch.load(pt_file, map_location="cpu")
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        total_params = 0
        
        for ckpt_key, ckpt_value in state_dict.items():
            matched_key = self._find_matching_key(ckpt_key, model_state_dict)
            
            if matched_key:
                # 形状チェック
                if model_state_dict[matched_key].shape != ckpt_value.shape:
                    print(f"   ⚠️ Shape mismatch for {matched_key}")
                    continue
                
                model_state_dict[matched_key] = ckpt_value.to(device=self.device, dtype=self.dtype)
                loaded_keys.append(matched_key)
                total_params += ckpt_value.numel()
        
        if len(loaded_keys) > 0:
            self.model.load_state_dict(model_state_dict, strict=False)
            print(f"   ✅ Loaded legacy .pt weights")
            print(f"   - Loaded keys: {len(loaded_keys)}")
            print(f"   - Total parameters: {total_params:,}")
        else:
            print(f"   ⚠️ WARNING: No parameters were loaded from .pt file!")

    def _find_matching_key(self, ckpt_key: str, model_state_dict: dict) -> Optional[str]:
        """
        チェックポイントのキーをモデルのキーにマッチング
        
        Args:
            ckpt_key: チェックポイント内のキー
            model_state_dict: モデルのstate_dict
        
        Returns:
            マッチしたモデルのキー（見つからない場合はNone）
        """
        # 1. 完全一致
        if ckpt_key in model_state_dict:
            return ckpt_key
        
        # 2. プレフィックス除去
        prefixes_to_strip = [
            "module.",  # DDP
            "base_model.model.model.",
            "base_model.model.",
            "model.",
        ]
        
        for prefix in prefixes_to_strip:
            if ckpt_key.startswith(prefix):
                stripped_key = ckpt_key[len(prefix):]
                
                # そのまま試す
                if stripped_key in model_state_dict:
                    return stripped_key
                
                # "model."を補って試す
                candidate = f"model.{stripped_key}"
                if candidate in model_state_dict:
                    return candidate
        
        # 3. 部分一致（サフィックスマッチング）
        for model_key in model_state_dict.keys():
            if model_key.endswith(ckpt_key) or ckpt_key.endswith(model_key):
                return model_key
        
        return None
    
    def initialize(self) -> bool:
        """
        モデルを初期化（起動時に一度だけ実行）
        
        Returns:
            bool: 初期化が成功したかどうか
        """
        if self.is_initialized:
            print("⚠️ Model already initialized")
            return True
        
        try:
            print("\n" + "="*80)
            print("Initializing PointLLM Model (Encoder + Proj + LoRA/LLM)")
            print("="*80)
            
            # デバイス設定
            self.device = get_device(verbose=True)
            
            # データ型設定
            if self.device.type == "cuda":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.device.type == "mps":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float16
            
            print(f"[INFO] Using dtype: {self.dtype}")
            
            # PointLLM環境のセットアップ
            if not setup_pointllm_environment():
                raise RuntimeError("Failed to setup PointLLM environment")
            
            # ステップ1: ベースモデルのロード
            self._load_base_model()
            
            # ステップ2: [オプション] Stage 1の重み（Encoder + Projector）をロード
            if self.training_mode in ["two_stage_lora", "two_stage_llm", "encoder_proj_only"]:
                self._load_encoder_projector_weights()
            else:
                print("\n[Stage 1] Skipped: Using frozen Encoder from base model")
            
            # ステップ3: Stage 2の重みをロード
            if self.training_mode == "two_stage_llm":
                # LLMファインチューニング済み重みをロード
                self._load_llm_finetuned_weights()
                
                # Projector重みをロード（LLM重みの後）
                if 'point_proj' in self.llm_files:
                    self._load_projector_update(self.llm_files['point_proj'])
                    
            elif self.training_mode in ["two_stage_lora", "proj_lora_only"]:
                # Projector重みをロード（LoRA適用の前）
                if 'point_proj' in self.lora_files:
                    self._load_projector_update(self.lora_files['point_proj'])
                
                # LoRAアダプタをロード
                self._load_lora_adapter()
            
            # 評価モードに設定
            self.model.eval()
            
            self.is_initialized = True
            print("\n" + "="*80)
            print("✅ Model initialized successfully")
            print(f"   Training Mode: {self.training_mode}")
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to initialize model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
            return False
    
    def get_model_info(self) -> dict:
        """モデル情報を取得"""
        return {
            "is_initialized": self.is_initialized,
            "training_mode": self.training_mode,
            "device": str(self.device) if self.device else None,
            "dtype": str(self.dtype) if self.dtype else None,
            "model_path": self.model_path,
            "encoder_checkpoint_path": self.encoder_checkpoint_path,
            "lora_checkpoint_path": self.lora_checkpoint_path,
            "llm_checkpoint_path": self.llm_checkpoint_path,
            "encoder_files_found": list(self.encoder_files.keys()),
            "lora_files_found": list(self.lora_files.keys()),
            "llm_files_found": list(self.llm_files.keys())
        }
    
    def check_lora_loaded(self) -> bool:
        """LoRAアダプタがロードされているか確認"""
        if not self.is_initialized:
            return False
        
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel)
        except ImportError:
            return False
    
    def check_encoder_loaded(self) -> bool:
        """Encoderがロードされているか確認"""
        if not self.is_initialized:
            return False
        
        try:
            for name, _ in self.model.named_parameters():
                if "point_backbone" in name:
                    return True
            return False
        except:
            return False
    
    def check_point_proj_loaded(self) -> bool:
        """Point Projectorがロードされているか確認"""
        if not self.is_initialized:
            return False
        
        try:
            for name, _ in self.model.named_parameters():
                if "point_proj" in name:
                    return True
            return False
        except:
            return False
    
    def get_detailed_info(self) -> dict:
        """詳細なモデル情報を取得"""
        info = self.get_model_info()
        
        if self.is_initialized:
            info.update({
                "encoder_loaded": self.check_encoder_loaded(),
                "point_proj_loaded": self.check_point_proj_loaded(),
                "lora_loaded": self.check_lora_loaded(),
                "model_type": type(self.model).__name__,
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
        
        return info
    
    def print_summary(self):
        """モデルサマリーを表示"""
        if not self.is_initialized:
            print("⚠️ Model not initialized yet")
            return
        
        print("\n" + "="*80)
        print("Model Summary")
        print("="*80)
        
        info = self.get_detailed_info()
        
        print(f"Training Mode: {info['training_mode']}")
        print(f"Model Type: {info['model_type']}")
        print(f"Device: {info['device']}")
        print(f"Dtype: {info['dtype']}")
        print("\nComponents Loaded:")
        print(f"  - Encoder: {'✅' if info['encoder_loaded'] else '❌'}")
        print(f"  - Projector: {'✅' if info['point_proj_loaded'] else '❌'}")
        print(f"  - LoRA: {'✅' if info['lora_loaded'] else '❌'}")
        print(f"\nParameters:")
        print(f"  - Total: {info['total_parameters']:,}")
        print(f"  - Trainable: {info['trainable_parameters']:,}")
        print("\nCheckpoint Files:")
        if info['encoder_files_found']:
            print(f"  - Encoder: {', '.join(info['encoder_files_found'])}")
        if info['lora_files_found']:
            print(f"  - LoRA: {', '.join(info['lora_files_found'])}")
        if info['llm_files_found']:
            print(f"  - LLM Finetune: {', '.join(info['llm_files_found'])}")
        print("="*80 + "\n")
    
    def __repr__(self) -> str:
        """文字列表現"""
        status = "Initialized" if self.is_initialized else "Not Initialized"
        components = []
        if self.is_initialized:
            if self.check_encoder_loaded():
                components.append("Encoder")
            if self.check_point_proj_loaded():
                components.append("Proj")
            if self.check_lora_loaded():
                components.append("LoRA")
        
        components_str = "+".join(components) if components else "None"
        mode_str = f", mode={self.training_mode}" if self.training_mode else ""
        return f"ModelManager(status={status}, components={components_str}{mode_str}, device={self.device})"


# 使用例
if __name__ == "__main__":
    # パターン1: 2段階学習（Encoder+Proj → Proj+LoRA）
    manager_two_stage_lora = ModelManager(
        model_path="RunsenXu/PointLLM_7B_v1.2",
        encoder_checkpoint_path="outputs/encoder_proj_finetune_v1/best_model",
        lora_checkpoint_path="outputs/encoder_proj_lora_finetune_v1/best_model"
    )
    
    # パターン2: Proj+LoRAのみ（Encoder凍結）
    manager_proj_lora = ModelManager(
        model_path="RunsenXu/PointLLM_7B_v1.2",
        encoder_checkpoint_path=None,
        lora_checkpoint_path="outputs/projection_lora_finetune_v2/best_model"
    )
    
    # パターン3: 2段階学習（Encoder+Proj → Proj+LLM Finetune）★新規追加
    manager_two_stage_llm = ModelManager(
        model_path="RunsenXu/PointLLM_7B_v1.2",
        encoder_checkpoint_path="outputs/encoder_proj_finetune_v5/best_model",
        llm_checkpoint_path="outputs/dental_model_head_coord_128_no_trash_encoder_proj_llm_finetune_v3/checkpoints/checkpoint-epoch-10"
    )
    
    # モデルのロード
    if manager_two_stage_llm.initialize():
        manager_two_stage_llm.print_summary()