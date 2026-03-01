#!/usr/bin/env python3
"""
validate_checkpoint.py - チェックポイントディレクトリの検証

使用方法:
    python validate_checkpoint.py <checkpoint_dir>
    
例:
    python validate_checkpoint.py pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple


def format_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読みやすい形式に変換"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """
    ファイルの存在確認
    
    Returns:
        (exists, size_info): 存在するか、サイズ情報
    """
    if filepath.exists():
        size = filepath.stat().st_size
        return True, format_size(size)
    return False, "Not found"


def validate_adapter_config(config_path: Path) -> Dict[str, any]:
    """
    adapter_config.jsonの検証
    
    Returns:
        Dict: 設定情報
    """
    if not config_path.exists():
        return {"error": "File not found"}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "peft_type": config.get("peft_type", "Unknown"),
            "task_type": config.get("task_type", "Unknown"),
            "r": config.get("r", "Unknown"),
            "lora_alpha": config.get("lora_alpha", "Unknown"),
            "lora_dropout": config.get("lora_dropout", "Unknown"),
            "target_modules": config.get("target_modules", []),
        }
    except Exception as e:
        return {"error": str(e)}


def validate_checkpoint_directory(checkpoint_dir: str) -> None:
    """
    チェックポイントディレクトリを検証して結果を表示
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
    """
    checkpoint_path = Path(checkpoint_dir)
    
    print("=" * 80)
    print("PointLLM Checkpoint Validation")
    print("=" * 80)
    print(f"Directory: {checkpoint_path.absolute()}")
    print()
    
    # ディレクトリの存在確認
    if not checkpoint_path.exists():
        print("❌ Error: Directory does not exist!")
        return
    
    if not checkpoint_path.is_dir():
        print("❌ Error: Path is not a directory!")
        return
    
    print("✅ Directory exists")
    print()
    
    # 必須ファイルのチェック
    print("-" * 80)
    print("Required Files Check")
    print("-" * 80)
    
    files_to_check = {
        "adapter_config.json": checkpoint_path / "adapter_config.json",
        "adapter_model.bin": checkpoint_path / "adapter_model.bin",
        "adapter_model.safetensors": checkpoint_path / "adapter_model.safetensors",
        "point_proj.pt": checkpoint_path / "point_proj.pt",
    }
    
    results = {}
    for name, filepath in files_to_check.items():
        exists, size_info = check_file_exists(filepath)
        results[name] = (exists, size_info)
        
        status = "✅" if exists else "❌"
        print(f"{status} {name:30s} {size_info}")
    
    print()
    
    # LoRAアダプタのチェック
    print("-" * 80)
    print("LoRA Adapter Check")
    print("-" * 80)
    
    has_adapter_config = results["adapter_config.json"][0]
    has_adapter_bin = results["adapter_model.bin"][0]
    has_adapter_safetensors = results["adapter_model.safetensors"][0]
    
    if has_adapter_config and (has_adapter_bin or has_adapter_safetensors):
        print("✅ LoRA adapter files found")
        
        # adapter_config.jsonの内容を表示
        config_info = validate_adapter_config(files_to_check["adapter_config.json"])
        
        if "error" not in config_info:
            print("\nAdapter Configuration:")
            print(f"  PEFT Type:       {config_info['peft_type']}")
            print(f"  Task Type:       {config_info['task_type']}")
            print(f"  Rank (r):        {config_info['r']}")
            print(f"  Alpha:           {config_info['lora_alpha']}")
            print(f"  Dropout:         {config_info['lora_dropout']}")
            print(f"  Target Modules:  {', '.join(config_info['target_modules'])}")
        else:
            print(f"⚠️ Warning: Could not parse adapter_config.json: {config_info['error']}")
    else:
        print("❌ LoRA adapter files incomplete")
        if not has_adapter_config:
            print("   Missing: adapter_config.json")
        if not has_adapter_bin and not has_adapter_safetensors:
            print("   Missing: adapter_model.bin or adapter_model.safetensors")
    
    print()
    
    # Point Projectorのチェック
    print("-" * 80)
    print("Point Projector Check")
    print("-" * 80)
    
    has_point_proj = results["point_proj.pt"][0]
    
    if has_point_proj:
        print("✅ Point Projector file found")
        print(f"   Size: {results['point_proj.pt'][1]}")
    else:
        print("❌ Point Projector file not found")
    
    print()
    
    # その他のファイルをリスト
    print("-" * 80)
    print("Other Files in Directory")
    print("-" * 80)
    
    other_files = []
    for item in checkpoint_path.iterdir():
        if item.is_file() and item.name not in files_to_check.keys():
            size = format_size(item.stat().st_size)
            other_files.append((item.name, size))
    
    if other_files:
        for name, size in sorted(other_files):
            print(f"   {name:40s} {size}")
    else:
        print("   (No other files)")
    
    print()
    
    # 総合評価
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_good = (
        has_adapter_config and
        (has_adapter_bin or has_adapter_safetensors) and
        has_point_proj
    )
    
    if all_good:
        print("✅ All required files are present!")
        print("   This checkpoint can be used for LoRA inference.")
    else:
        print("⚠️ Some required files are missing!")
        print("   This checkpoint may not work correctly.")
        
        if not has_adapter_config:
            print("\n   Required: adapter_config.json")
        if not has_adapter_bin and not has_adapter_safetensors:
            print("\n   Required: adapter_model.bin OR adapter_model.safetensors")
        if not has_point_proj:
            print("\n   Required: point_proj.pt")
    
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_checkpoint.py <checkpoint_dir>")
        print()
        print("Example:")
        print("  python validate_checkpoint.py pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    validate_checkpoint_directory(checkpoint_dir)


if __name__ == "__main__":
    main()
