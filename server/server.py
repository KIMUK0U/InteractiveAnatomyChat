"""
server.py - Mac mini側のメインサーバー（LoRA + LLM Finetune対応版 + メモリリーク対策 + 計測機能付き）

Vision ProからのトラッキングデータをPointLLMで処理
LoRAアダプタ、LLMファインチューニング、Point Projectorの全てに対応
・入力点群の可視化保存機能を追加
・メモリリーク対策を追加
・★処理時間計測とログ保存機能を追加
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os
import matplotlib
matplotlib.use('Agg')  # GUIなしバックエンドを使用
import matplotlib.pyplot as plt
import gc
import torch
from scipy.spatial.transform import Rotation as R_scipy
import time  # ★追加: 時間計測用
import csv   # ★追加: ログ保存用
import psutil  # 既存

# PointLLM関連のモジュールをインポートするための設定
POINTLLM_ROOT = Path(__file__).parent / "PointLLM"
sys.path.insert(0, str(POINTLLM_ROOT))
os.chdir(POINTLLM_ROOT)

from pointllm_manager import (
    ModelManager,
    generate_response_with_pointllm,
    validate_point_cloud
)

from pc_process import (
    find_point_cloud_file,
    load_point_cloud,
    convert_pc_from_ar_data,
    process_and_analyze_interaction,
    transform_pc_to_head_space,
    normalize_point_cloud
)

app = FastAPI(
    title="PointLLM Vision Pro Server",
    description="Vision ProからのトラッキングデータをPointLLMで処理（LoRA + LLM Finetune対応 + メモリ最適化 + 計測）",
    version="1.2.0-llm-perf"
)

# ===========================
# データモデル定義 (変更なし)
# ===========================
# ... (元のコードと同じため省略) ...
class Vector3(BaseModel):
    x: float
    y: float
    z: float

class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float

class JointData(BaseModel):
    jointName: str
    position: Vector3
    rotation: Quaternion
    isTracked: bool
    localPosition: Optional[Vector3] = None
    localRotation: Optional[Quaternion] = None

class HandTransform(BaseModel):
    position: Vector3
    rotation: Quaternion

class HandData(BaseModel):
    isTracked: bool
    chirality: str
    handTransform: Optional[HandTransform] = None
    joints: List[JointData]

class DeviceTransform(BaseModel):
    position: Vector3
    forward: Vector3
    up: Vector3
    right: Vector3
    rotation: Quaternion

class ObjectData(BaseModel):
    objectID: str
    objectName: str
    modelFileName: str
    position: Vector3
    forward: Vector3
    up: Vector3
    right: Vector3
    rotation: Quaternion
    scale: Vector3

class FrameData(BaseModel):
    timestamp: float
    frameNumber: int
    deviceTransform: Optional[DeviceTransform] = None
    leftHand: Optional[HandData] = None
    rightHand: Optional[HandData] = None
    objects: List[ObjectData]

class SessionMetadata(BaseModel):
    deviceModel: str
    osVersion: str
    appVersion: str
    markerColor: Optional[str] = "red"

class TrackingSession(BaseModel):
    sessionStartTime: str
    sessionEndTime: Optional[str] = None
    frames: List[FrameData]
    metadata: SessionMetadata

class TrackingRequest(BaseModel):
    session: TrackingSession
    question: str


# ===========================
# グローバル変数（モデル管理）
# ===========================

# ModelManagerのインスタンス（LoRA + LLM Finetune対応版）
model_manager = ModelManager(
    model_path="RunsenXu/PointLLM_7B_v1.2",
    encoder_checkpoint_path=str(POINTLLM_ROOT / "pointllm/outputs/encoder_proj_tune_ddp_demo_ar_410_v1/checkpoints/checkpoint-epoch-20"),
    llm_checkpoint_path=str(POINTLLM_ROOT / "pointllm/outputs/demo_410_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v4_EPLR_high_30epoch/checkpoints/best_model_epoch14_mashi"),
)

# 設定
_SERVER_DIR = Path(__file__).parent
USDZ_PC_DIR = _SERVER_DIR / "data/usdz_pc"
CLASS_INFO_DIR = _SERVER_DIR / "data/class_info"
DEBUG_DIR = Path("debug_data")
DEBUG_DIR.mkdir(exist_ok=True)

# 結果画像保存用ディレクトリ
RESULTS_DIR = _SERVER_DIR / "data/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ★追加: パフォーマンスログ保存用設定
LOG_DIR = _SERVER_DIR / "data/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_LOG_FILE = LOG_DIR / "performance_metrics.csv"

# server.py の log_performance_metrics を以下に差し替え
def log_performance_metrics(
    processing_time: float,
    inference_time: float,
    total_time: float,
    num_points: int,
    question: str,
    structure_detected: str,
    memory_info: Dict[str, float],
    inference_details: Dict[str, float] = None # ★詳細メモリ用に追加
):
    """
    計測結果と推論中の詳細メモリ使用量をCSVに保存
    """
    try:
        file_exists = PERFORMANCE_LOG_FILE.exists()
        
        with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "process_mem_mb",
                    "mps_idle_mb",         # 待機時
                    "mps_inference_start",  # 推論開始時
                    "mps_point_to_device",  # 点群転送後
                    "mps_generation_end",   # 生成終了時（最大）
                    "processing_time_sec",
                    "inference_time_sec",
                    "total_time_sec",
                    "num_points",
                    "question",
                    "detected_structure"
                ])
            
            # 詳細データがない場合のフォールバック
            details = inference_details or {}
            
            writer.writerow([
                datetime.now().isoformat(),
                f"{memory_info.get('process_rss_mb', 0):.2f}",
                f"{memory_info.get('mps_driver_mb', 0):.2f}",
                f"{details.get('start', 0):.2f}",
                f"{details.get('point_loaded', 0):.2f}",
                f"{details.get('end', 0):.2f}",
                f"{processing_time:.4f}",
                f"{inference_time:.4f}",
                f"{total_time:.4f}",
                num_points,
                question.replace('\n', ' '),
                structure_detected.replace('\n', ' ')
            ])
        print(f"📝 Detailed Metrics logged to: {PERFORMANCE_LOG_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to log metrics: {e}")

# ===========================
# メモリ管理ユーティリティ (変更なし)
# ===========================

def clear_memory(device: torch.device = None):
    # ... (変更なし)
    gc.collect()
    if device is not None:
        if device.type == 'mps':
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception:
                pass
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

def get_memory_usage(device: torch.device = None) -> Dict[str, float]:
    """
    現在のプロセス（このプログラム）単体のメモリ使用量を取得
    
    Returns:
        Dict: メモリ使用量の辞書
    """
    # 現在のプロセスを取得
    process = psutil.Process(os.getpid())
    
    # RSS (Resident Set Size): 物理メモリ使用量
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 ** 2)  # MB単位
    
    result = {
        "process_rss_mb": rss_mb,  # プロセス自体のメモリ使用量
        "system_percent": psutil.virtual_memory().percent, # 参考: システム全体の使用率
    }
    
    # M4 Pro (MPS) 用のメモリ計測
    if device is not None and device.type == 'mps':
        try:
            # MPSが現在確保しているメモリ
            allocated = torch.mps.current_allocated_memory() / (1024**2)
            # MPSドライバが確保しているメモリ（OS管理分含む）
            driver = torch.mps.driver_allocated_memory() / (1024**2)
            
            result["mps_allocated_mb"] = allocated
            result["mps_driver_mb"] = driver
        except Exception:
            pass
            
    return result

# ===========================
# ユーティリティ関数 (変更なし)
# ===========================

def create_question_from_hand_data(frame_data: FrameData, user_question: str) -> str:
    if user_question and user_question.strip():
        return user_question.strip()
    return ""

def save_debug_image(point_cloud: Any, filename_prefix: str = "pc_vis") -> Optional[str]:
    # ... (変更なし)
    fig = None
    try:
        if hasattr(point_cloud, 'cpu'):
            pc_np = point_cloud.cpu().numpy()
        elif isinstance(point_cloud, np.ndarray):
            pc_np = point_cloud.copy()
        else:
            return None

        if pc_np.shape[0] > 10000:
            indices = np.random.choice(pc_np.shape[0], 10000, replace=False)
            pc_np = pc_np[indices]

        xyz = pc_np[:, :3]
        rgb = pc_np[:, 3:6]
        rgb = np.clip(rgb, 0, 1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=2, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=90, azim=-90)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RESULTS_DIR / f"{filename_prefix}_{timestamp}.png"
        
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        del pc_np, xyz, rgb
        return str(filename)
    except Exception as e:
        print(f"⚠️ Failed to save debug image: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close('all')

# ===========================
# メイン処理関数（計測機能追加版）
# ===========================

async def process_with_pointllm(
    data: TrackingSession,
    question: str
) -> Dict[str, Any]:
    """Vision Proから受け取ったデータをPointLLMで処理（計測機能付き）"""
    
    # 全体の開始
    start_total_time = time.perf_counter()
    t_ref = start_total_time # ステップ計測用の基準時間

    def mark_time(label, prev_t):
        now = time.perf_counter()
        duration = now - prev_t
        print(f"⏱️  [Step] {label}: {duration:.4f} sec")
        return now
    
    # ローカル変数の初期化
    point_cloud = None
    converted_pc = None
    colored_pc = None
    convert_head_pc = None
    final_input_pc = None
    answer = None
    rotation_matrix = None
    
    # 計測変数の初期化
    time_processing_sec = 0.0
    time_inference_sec = 0.0
    
    try:
        # 1. モデルの初期化確認
        if not model_manager.is_initialized:
            if not model_manager.initialize():
                raise RuntimeError("Failed to initialize model")
        t_ref = mark_time("Model Init Check", t_ref)

        # 2. フレームデータの取得
        if not data.frames:
            raise ValueError("No frames in tracking session")
        frame = data.frames[0]
        
        if not frame.objects:
            raise ValueError("No 3D objects in frame")
        obj = frame.objects[0]
        t_ref = mark_time("Data Extraction", t_ref)

        # 3. ファイル検索と回転行列の算出
        if hasattr(obj, 'rotation') and obj.rotation is not None:
            quat = [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w]
            rot = R_scipy.from_quat(quat)
            rotation_matrix = rot.as_matrix()
        else:
            rotation_matrix = np.eye(3)
        
        pc_file = find_point_cloud_file(obj.modelFileName, USDZ_PC_DIR)
        t_ref = mark_time("File Search & Rotation Calc", t_ref)

        # 4. 点群データの読み込み (I/O)
        point_cloud = load_point_cloud(pc_file)
        if not validate_point_cloud(point_cloud):
            raise ValueError("Invalid point cloud data")
        t_ref = mark_time("Point Cloud Load (I/O)", t_ref)

        # 5. 座標変換（AR Data -> Object Space）
        converted_pc = convert_pc_from_ar_data(
            point_cloud=point_cloud,
            object_data=obj,
            usdz_path=None
        )
        del point_cloud; point_cloud = None
        t_ref = mark_time("Transform: AR to Object", t_ref)

        # 6. 点群の彩色とインタラクション解析 (k-d tree等を使用)
        target_marker_color = data.metadata.markerColor if data.metadata.markerColor else "red"
        
        # クラス情報のロード時間は微小なため結合
        usdz_stem = Path(obj.modelFileName).stem
        class_info_path = CLASS_INFO_DIR / f"{usdz_stem}_class_subclass_info.json"
        class_info_data = None
        if class_info_path.exists():
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info_data = json.load(f)

        colored_pc, analysis_json = process_and_analyze_interaction(
            point_cloud=converted_pc,
            frame_data=frame,
            target_color=target_marker_color,
            class_info=class_info_data
        )
        del converted_pc; converted_pc = None
        t_ref = mark_time("Interaction Analysis (k-d tree)", t_ref)

        # 7. 座標変換（World/Head Space & Normalize）
        convert_head_pc = transform_pc_to_head_space(
            point_cloud=colored_pc,
            device_transform=frame.deviceTransform
        )
        del colored_pc; colored_pc = None
        
        final_input_pc = normalize_point_cloud(convert_head_pc)
        del convert_head_pc; convert_head_pc = None
        t_ref = mark_time("Transform: Head Space & Normalize", t_ref)

        # 8. 可視化保存 (File Write)
        image_path = save_debug_image(final_input_pc, filename_prefix="pointllm_input")
        t_ref = mark_time("Save Debug Image", t_ref)

        # 9. 推論用コンテキスト作成
        final_question = create_question_from_hand_data(frame, question)
        from pointllm_manager import create_system_context_from_rotation
        dynamic_system_context = create_system_context_from_rotation(
            rotation_matrix, lang="en", task_type="color_identification"
        )
        t_ref = mark_time("Context Preparation", t_ref)

        # 計測終了（推論前まで）
        time_processing_sec = time.perf_counter() - start_total_time
        print(f"✅ Total Pre-processing Time: {time_processing_sec:.4f} sec")

        # ★計測: (c) 推論開始
        start_inference_time = time.perf_counter()
        # --- server.py の推論呼び出し直前 ---
        if model_manager.device.type == 'mps':
            print(f"🔥 BEFORE INFERENCE: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
        answer = generate_response_with_pointllm(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            point_cloud=final_input_pc,
            question=final_question,
            device=model_manager.device,
            usdz_filename=obj.modelFileName,
            class_info_path=str(class_info_path),
            target_color=target_marker_color,
            hand_distance_mm=analysis_json.get("min_distance_mm", 0.0),
            system_context=dynamic_system_context,
            lang="en",
            max_new_tokens=512,
            temperature=0.5
        )
        if model_manager.device.type == 'mps':
            print(f"🔥 AFTER INFERENCE: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
        # ★計測: (c) 推論終了
        time_inference_sec = time.perf_counter() - start_inference_time
        print(f"⏱️ (c) Inference Time: {time_inference_sec:.4f} sec")
        print(f"💡 Answer: {answer}")

        # ★計測: 合計時間
        total_time_sec = time.perf_counter() - start_total_time

        # ★計測: 推論直後のメモリ状態を取得（ここがピークメモリに近い）
        current_mem = get_memory_usage(model_manager.device if model_manager.is_initialized else None)
        print(f"🧠 Current Process Memory: {current_mem.get('process_rss_mb', 0):.1f} MB (MPS: {current_mem.get('mps_allocated_mb', 0):.1f} MB)")

        # ★ログ保存 (引数にメモリ情報を追加)
        log_performance_metrics(
            processing_time=time_processing_sec,
            inference_time=time_inference_sec,
            total_time=total_time_sec,
            num_points=len(final_input_pc),
            question=final_question,
            structure_detected=answer,
            memory_info=current_mem  # ★追加
        )
        
        # 8. 結果を返す（計測情報も含める）
        result = {
            "detected_structure": answer,
            "question": final_question,
            "point_cloud_file": str(pc_file),
            "num_points": len(final_input_pc),
            "hand_tracking": {
                "left_hand_tracked": frame.leftHand is not None and frame.leftHand.isTracked,
                "right_hand_tracked": frame.rightHand is not None and frame.rightHand.isTracked
            },
            "interaction_analysis": analysis_json,
            "debug_image": image_path,
            "model_info": {
                "lora_loaded": model_manager.check_lora_loaded(),
                "point_proj_loaded": model_manager.check_point_proj_loaded()
            },
            # ★追加: クライアント側へ時間を返す
            "performance_metrics": {
                "processing_time_sec": time_processing_sec,
                "inference_time_sec": time_inference_sec,
                "total_time_sec": total_time_sec
            }
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Error in process_with_pointllm: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        del point_cloud, converted_pc, colored_pc, convert_head_pc, final_input_pc, answer
        clear_memory(model_manager.device if model_manager.is_initialized else None)

# ===========================
# APIエンドポイント (変更なし)
# ===========================
# ... (Startup, health_checkなどは変更なし) ...

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*80)
    print("Starting PointLLM Server (LoRA + LLM Finetune + Memory Optimized + Perf Log)...")
    print("="*80)
    success = model_manager.initialize()
    # ... (省略)
    print("="*80 + "\n")

@app.get("/api/health")
async def health_check():
    # ... (省略)
    return {
        "status": "ok",
        "server": "Mac mini",
        "version": "1.2.0-llm-perf", # バージョン更新
        "model_initialized": model_manager.is_initialized,
        "lora_loaded": model_manager.check_lora_loaded() if model_manager.is_initialized else False,
        "point_proj_loaded": model_manager.check_point_proj_loaded() if model_manager.is_initialized else False,
        "memory": get_memory_usage(model_manager.device if model_manager.is_initialized else None)
    }

@app.get("/api/model_info")
async def get_model_info():
    return model_manager.get_detailed_info()

@app.post("/api/process_tracking")
async def process_tracking(request: TrackingRequest):
    try:
        result = await process_with_pointllm(request.session, request.question)
        return {
            "status": "success",
            "processed_at": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/debug/save_data")
async def save_debug_data(request: TrackingRequest):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = DEBUG_DIR / f"tracking_request_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(request.dict(), f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "saved_to": str(filename)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/simple")
async def test_simple():
    return {
        "status": "ok",
        "message": "Server is running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/clear_cache")
async def clear_cache_endpoint():
    try:
        mem_before = get_memory_usage(model_manager.device if model_manager.is_initialized else None)
        clear_memory(model_manager.device if model_manager.is_initialized else None)
        mem_after = get_memory_usage(model_manager.device if model_manager.is_initialized else None)
        
        return {
            "status": "success",
            "memory_before": mem_before,
            "memory_after": mem_after,
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PointLLM Vision Pro Server (LoRA + LLM Finetune + Perf Log)")
    print("="*80)
    # ... (省略)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )