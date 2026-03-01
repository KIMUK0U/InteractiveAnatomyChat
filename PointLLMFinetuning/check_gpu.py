#!/usr/bin/env python3
"""
GPU使用状況確認スクリプト

使用方法:
    python check_gpu.py
"""

import torch
import subprocess
import sys

def check_cuda_available():
    """CUDAの利用可能性をチェック"""
    print("=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        print("PyTorch was not built with CUDA support")
        sys.exit(1)
    
    print("✅ CUDA is available")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

def check_gpu_details():
    """GPU詳細情報の表示"""
    print("=" * 60)
    print("GPU Details")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    print()
    
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        
        # メモリ使用状況
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        
        print(f"  Memory Allocated: {mem_allocated:.2f} GB")
        print(f"  Memory Reserved: {mem_reserved:.2f} GB")
        print(f"  Memory Free: {mem_free:.2f} GB")
        print()

def check_nvidia_smi():
    """nvidia-smiコマンドで詳細確認"""
    print("=" * 60)
    print("nvidia-smi Output")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            idx, name, mem_used, mem_total, util = line.split(', ')
            mem_used_gb = float(mem_used) / 1024
            mem_total_gb = float(mem_total) / 1024
            mem_free_gb = mem_total_gb - mem_used_gb
            
            print(f"GPU {idx}: {name}")
            print(f"  Memory: {mem_used_gb:.2f} GB / {mem_total_gb:.2f} GB used ({mem_free_gb:.2f} GB free)")
            print(f"  GPU Utilization: {util}%")
            print()
            
    except subprocess.CalledProcessError:
        print("⚠️ nvidia-smi command failed")
    except FileNotFoundError:
        print("⚠️ nvidia-smi not found in PATH")

def recommend_gpu():
    """使用推奨GPUを提案"""
    print("=" * 60)
    print("Recommended GPU for Training")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("❌ No GPU available")
        return
    
    # 各GPUのメモリ空き容量を計算
    gpu_info = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = mem_total - mem_reserved
        
        gpu_info.append({
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'mem_free': mem_free,
            'mem_total': mem_total
        })
    
    # 空きメモリでソート
    gpu_info.sort(key=lambda x: x['mem_free'], reverse=True)
    
    print("GPUs sorted by available memory:")
    for i, info in enumerate(gpu_info):
        status = "✅ RECOMMENDED" if i == 0 else "  "
        print(f"{status} GPU {info['index']}: {info['name']}")
        print(f"     Free: {info['mem_free']:.2f} GB / {info['mem_total']:.2f} GB")
    
    print()
    recommended_gpu = gpu_info[0]['index']
    print(f"💡 Use GPU {recommended_gpu} for training")
    print(f"   Set environment variable: CUDA_VISIBLE_DEVICES={recommended_gpu}")
    print(f"   Or in Python: torch.cuda.set_device({recommended_gpu})")
    print()

def test_gpu_tensor():
    """簡単なテンソル演算でGPUの動作確認"""
    print("=" * 60)
    print("GPU Tensor Operation Test")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        try:
            device = torch.device(f'cuda:{i}')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            
            print(f"✅ GPU {i}: Tensor operations successful")
        except Exception as e:
            print(f"❌ GPU {i}: Failed - {e}")
    print()

if __name__ == "__main__":
    try:
        check_cuda_available()
        check_gpu_details()
        check_nvidia_smi()
        test_gpu_tensor()
        recommend_gpu()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
