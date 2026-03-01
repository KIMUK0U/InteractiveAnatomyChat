"""
PointLLMでの推論実行モジュール(LoRA対応版) - 動的LPS軸更新対応版
"""

import torch
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Dict
import json

# PointLLMの既存関数をインポート
from pointllm.eval.chat_gradio_finetuning_encoder_proj_LoRA import generate_response

# ==============================================================================
# データセット作成用モジュールのインポート
# ==============================================================================
try:
    from .modules.anatomical_naming import AnatomicalNaming
    print("✅ AnatomicalNaming loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: AnatomicalNaming import failed: {e}")
    AnatomicalNaming = None

# ==============================================================================
# 動的LPS軸更新用モジュールのインポート
# ==============================================================================
try:
    from .modules.head_space_lps_analyzer import HeadSpaceLPSAnalyzer, LPSDirections
    print("✅ HeadSpaceLPSAnalyzer loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: HeadSpaceLPSAnalyzer import failed: {e}")
    print("   Dynamic LPS axis update will be disabled")
    HeadSpaceLPSAnalyzer = None
    LPSDirections = None


# ==============================================================================
# 動的LPSコンテキスト生成関数
# ==============================================================================

def create_dynamic_lps_context(
    lps_directions: 'LPSDirections',
    lang: str = "en"
) -> str:
    """
    LPS方向情報から動的にシステムコンテキストを生成
    
    Args:
        lps_directions: Head座標系におけるLPS方向
        lang: 言語 ('en' or 'ja')
    
    Returns:
        動的に生成されたシステムコンテキスト
    """
    if LPSDirections is None or lps_directions is None:
        return get_default_system_context(lang)
    
    # 各LPS方向の主要軸と符号を取得
    left_axis, left_sign = lps_directions.get_dominant_axis('L')
    post_axis, post_sign = lps_directions.get_dominant_axis('P')
    sup_axis, sup_sign = lps_directions.get_dominant_axis('S')
    
    # 符号付き軸名を生成
    def signed_axis(axis: str, sign: float) -> str:
        return f"+{axis}" if sign > 0 else f"-{axis}"
    
    # L/R, P/A, S/Iそれぞれの表記
    L_repr = signed_axis(left_axis, left_sign)
    R_repr = signed_axis(left_axis, -left_sign)
    P_repr = signed_axis(post_axis, post_sign)
    A_repr = signed_axis(post_axis, -post_sign)
    S_repr = signed_axis(sup_axis, sup_sign)
    I_repr = signed_axis(sup_axis, -sup_sign)
    
    if lang == "en":
        return (
            "You are an AI assistant analyzing 3D dental anatomy in Head Space coordinate system. "
            "The data follows the LPS (Left-Posterior-Superior) coordinate system, defined relative to the patient:\n"
            f"- {left_axis}-axis: {L_repr} is the patient's Left (+L), {R_repr} is the patient's Right (-R).\n"
            f"- {post_axis}-axis: {P_repr} is Posterior (+P), {A_repr} is Anterior (-A).\n"
            f"- {sup_axis}-axis: {S_repr} is Superior (+S), {I_repr} is Inferior (-I).\n"
            "Please identify the anatomical structure highlighted in the specified color, considering this spatial orientation."
        )
    else:  # ja
        return (
            "あなたはHead Space座標系における3D歯科解剖構造を分析するAIアシスタントです。\n"
            "データは患者を基準としたLPS(左・後・上)座標系に従っています:\n"
            f"- {left_axis}軸:{L_repr}は患者の「左」(+L)、{R_repr}は患者の「右」(-R)\n"
            f"- {post_axis}軸:{P_repr}は「後方」(+P)、{A_repr}は「前方」(-A)\n"
            f"- {sup_axis}軸:{S_repr}は「上方」(+S)、{I_repr}は「下方」(-I)\n"
            "この空間的な方向定義に基づき、指定された色で強調表示されている解剖学的構造を特定してください。"
        )


def get_lps_directions_from_tracking(
    tracking_json_path: str,
    usdz_centroid: np.ndarray,
    object_id: str = "UserTargetModel"
) -> Optional['LPSDirections']:
    """
    TrackingDataからLPS方向情報を取得
    
    Args:
        tracking_json_path: TrackingData JSONファイルパス
        usdz_centroid: USDZモデル重心 (DICOM座標系, メートル)
        object_id: オブジェクトID
    
    Returns:
        LPS方向情報、取得失敗時はNone
    """
    if HeadSpaceLPSAnalyzer is None:
        print("⚠️ HeadSpaceLPSAnalyzer not available, using default context")
        return None
    
    try:
        _, lps_directions = HeadSpaceLPSAnalyzer.from_tracking_data(
            tracking_json_path,
            usdz_centroid,
            object_id
        )
        
        if lps_directions:
            print(f"✅ LPS directions loaded from: {Path(tracking_json_path).name}")
        else:
            print(f"⚠️ Failed to extract LPS from: {Path(tracking_json_path).name}")
        
        return lps_directions
    
    except Exception as e:
        print(f"⚠️ Error loading LPS directions: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# コンテキスト抽出関数（train.jsonからの静的抽出 - フォールバック用）
# ==============================================================================

def extract_system_context_from_train_json(train_json_path: str) -> Optional[str]:
    """
    train.jsonから最初のuser roleメッセージを抽出してシステムコンテキストとして使用
    
    Args:
        train_json_path: train.jsonのパス
    
    Returns:
        システムコンテキスト文字列（<point>タグを除く）
    """
    try:
        with open(train_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get('data'):
            return None
        
        # 最初のサンプルのconversationsを取得
        first_sample = data['data'][0]
        conversations = first_sample.get('conversations', [])
        
        # user roleのコンテンツを探す
        for conv in conversations:
            if conv.get('role') == 'user':
                content = conv.get('content', '')
                
                # <point>タグを除去してコンテキスト部分のみ抽出
                if '<point>' in content:
                    # <point>より前の部分を取得
                    context = content.split('<point>')[0].strip()
                    if context:
                        print(f"✅ Extracted system context from train.json ({len(context)} chars)")
                        return context
        
        return None
    
    except Exception as e:
        print(f"⚠️ Failed to extract context from train.json: {e}")
        return None


def get_default_system_context(lang: str = "en") -> str:
    """
    デフォルトのシステムコンテキスト（動的取得が失敗した場合）
    
    Args:
        lang: 言語 ('en' or 'ja')
    
    Returns:
        デフォルトのシステムコンテキスト
    """
    if lang == "en":
        return (
            "You are an AI assistant analyzing 3D dental anatomy. "
            "The data follows the LPS (Left-Posterior-Superior) coordinate system, defined relative to the patient:\n"
            "- X-axis: Positive is the patient's Left (+L), Negative is the patient's Right (-R).\n"
            "- Y-axis: Positive is Posterior (+P), Negative is Anterior (-A).\n"
            "- Z-axis: Positive is Superior (+S), Negative is Inferior (-I).\n"
            "This point cloud contains a color-marked region at the user-pointed location. When identifying anatomical structures, provide responses in the following format:\n1. Anatomical location description (using LPS directional terms)\n2. FDI notation for specific teeth (if applicable)\n\nExample responses:\n- 'Right Posterior Superior of Maxilla and Upper Skull'\n- 'FDI 24: Upper left 4th tooth from the midline (1st premolar)'\n- 'Left Posterior Inferior Buccal of Mandible'\n- 'FDI 35: Lower left 5th tooth from the midline (2nd premolar)'\n- 'FDI 12: Upper right 2nd tooth from the midline (Lateral incisor)'\n"
            "USER: "
        )
    else:  # ja
        return (
            "あなたは3D歯科解剖構造を分析するAIアシスタントです。\n"
            "データは患者を基準としたLPS(左・後・上)座標系に従っています:\n"
            "- X軸:正の方向は患者の「左」(+L)、負の方向は患者の「右」(-R)\n"
            "- Y軸:正の方向は「後方」(+P)、負の方向は「前方」(-A)\n"
            "- Z軸:正の方向は「上方」(+S)、負の方向は「下方」(-I)\n"
            "この空間的な方向定義に基づき、指定された色で強調表示されている解剖学的構造を特定してください。"
        )


# ==============================================================================
# ヘルパー関数
# ==============================================================================

def _get_colorization_description(target_color) -> dict:
    """
    マーカー色に基づいて説明文を生成
    
    Args:
        target_color: np.array([R, G, B]) または 'red', 'blue', 'black', 'white'
    """
    # 文字列の場合はRGB配列に変換
    if isinstance(target_color, str):
        color_map = {
            'red': np.array([1.0, 0.0, 0.0]),
            'blue': np.array([0.0, 0.0, 1.0]),
            'black': np.array([0.0, 0.0, 0.0]),
            'white': np.array([1.0, 1.0, 1.0])
        }
        target_color = color_map.get(target_color.lower(), np.array([1.0, 0.0, 0.0]))
    
    # RGB値で判定
    if np.allclose(target_color, [1.0, 0.0, 0.0]):
        return {'verb': 'highlighted in red', 'adjective': 'red colored'}
    elif np.allclose(target_color, [0.0, 0.0, 1.0]):
        return {'verb': 'highlighted in blue', 'adjective': 'blue colored'}
    elif np.allclose(target_color, [0.0, 0.0, 0.0]):
        return {'verb': 'marked in black', 'adjective': 'black colored'}
    elif np.allclose(target_color, [1.0, 1.0, 1.0]):
        return {'verb': 'marked in white', 'adjective': 'white colored'}
    else:
        return {'verb': 'highlighted', 'adjective': 'colored'}


def _infer_class_combination_from_filename(usdz_filename: str) -> str:
    """
    USDZファイル名からクラス組み合わせを推測
    
    実際の組み合わせ:
    - dental_model
    - Lower_Teeth
    - Upper_Teeth
    - U_and_LTeeth
    - U_L_and_Mandible
    - Skull_and_UTeeth
    - Mandible_and_LTeeth
    """
    stem = Path(usdz_filename).stem.lower()
    
    mapping = {
        "dental_model": "dental_model",
        "lower_teeth": "Lower_Teeth",
        "upper_teeth": "Upper_Teeth",
        "u_and_lteeth": "U_and_LTeeth",
        "u_l_and_mandible": "U_L_and_Mandible",
        "skull_and_uteeth": "Skull_and_UTeeth",
        "mandible_and_lteeth": "Mandible_and_LTeeth",
        # エイリアス
        "full_skull": "dental_model",
        "skull": "dental_model",
        "lower": "Lower_Teeth",
        "upper": "Upper_Teeth",
    }
    
    return mapping.get(stem, "dental_model")


# ==============================================================================
# メイン推論関数
# ==============================================================================

def generate_response_with_pointllm(
    model,
    tokenizer,
    point_cloud: np.ndarray,
    question: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    num_points: int = 8192,
    # 追加コンテキスト用
    usdz_filename: str = None,
    class_info_path: str = None,
    target_color: np.ndarray = None,
    hand_distance_mm: float = 0.0,
    # 動的LPS軸更新用（優先度: 最高）
    tracking_json_path: str = None,
    usdz_centroid: np.ndarray = None,
    object_id: str = "UserTargetModel",
    lang: str = "en",
    # 静的コンテキスト用（フォールバック）
    system_context: str = None,
    train_json_path: str = None
) -> str:
    """
    PointLLMで推論を実行（動的LPS軸更新対応）
    
    優先順位:
    1. tracking_json_path + usdz_centroid → 動的LPS軸更新
    2. system_context → 明示的に指定されたコンテキスト
    3. train_json_path → train.jsonから抽出
    4. デフォルトコンテキスト
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群データ (N, 3) or (N, 6)
        question: ユーザーの質問
        device: 使用デバイス
        max_new_tokens: 最大生成トークン数
        temperature: サンプリング温度
        top_p: nucleus sampling
        num_points: サンプリング後の点数
        usdz_filename: USDZファイル名
        class_info_path: クラス情報JSONへのパス
        target_color: マーカー色（np.array）
        hand_distance_mm: 手からの距離
        tracking_json_path: TrackingData JSONパス（動的LPS軸更新用）
        usdz_centroid: USDZモデル重心（動的LPS軸更新用、DICOM座標系、メートル）
        object_id: オブジェクトID（動的LPS軸更新用）
        lang: 言語 ('en' or 'ja')
        system_context: システムコンテキスト（明示的指定）
        train_json_path: train.jsonパス（コンテキスト自動抽出用）
    
    Returns:
        str: モデルの応答
    """
    try:
        # ========================================
        # システムコンテキストの取得（優先順位順）
        # ========================================
        context_prefix = ""
        lps_updated = False
        
        # 優先度1: 動的LPS軸更新
        if tracking_json_path and usdz_centroid is not None:
            print("🔄 Attempting dynamic LPS axis update...")
            lps_directions = get_lps_directions_from_tracking(
                tracking_json_path,
                usdz_centroid,
                object_id
            )
            
            if lps_directions:
                context_prefix = create_dynamic_lps_context(lps_directions, lang) + "\n"
                lps_updated = True
                print(f"✅ Using dynamic LPS context (lang={lang})")
            else:
                print("⚠️ Dynamic LPS update failed, falling back to static context")
        
        # 優先度2: 明示的なシステムコンテキスト
        if not lps_updated and system_context is not None:
            context_prefix = system_context + "\n"
            print("✅ Using explicit system context")
        
        # 優先度3: train.jsonから抽出
        if not lps_updated and not system_context and train_json_path:
            extracted_context = extract_system_context_from_train_json(train_json_path)
            if extracted_context:
                context_prefix = extracted_context + "\n"
                print("✅ Using context from train.json")
        
        # 優先度4: デフォルトコンテキスト
        if not context_prefix:
            context_prefix = get_default_system_context(lang) + "\n"
            print(f"✅ Using default system context (lang={lang})")
        
        # ========================================
        # 追加コンテキストの付与（既存機能）
        # ========================================
        additional_context = ""
        
        if usdz_filename and class_info_path and target_color is not None and AnatomicalNaming:
            try:
                # クラス組み合わせを推測
                class_combination = _infer_class_combination_from_filename(usdz_filename)
                
                # AnatomicalNamingを読み込み
                naming = AnatomicalNaming.from_json(class_info_path)
                
                # サブクラスリストを取得
                candidates = naming.get_filtered_subclass_display_names(class_combination)
                
                if candidates:
                    num_candidates = len(candidates)
                    all_candidates_str = ", ".join(candidates)
                    
                    # 色情報の取得
                    color_info = _get_colorization_description(target_color)
                    
                    # 追加コンテキスト文章を構築
                    additional_context = (
                        f"This dental CBCT model contains {num_candidates} anatomical structures: [{all_candidates_str}]. "
                        f"In the point cloud visualization, one region has been {color_info['verb']} "
                        f"to indicate where the user's hand (tracked in AR) is pointing.\n"
                    )
                    
                    print(f"✅ Additional context added: {usdz_filename} -> {class_combination}")
                else:
                    print(f"⚠️ No candidates for: {class_combination}")
                    
            except Exception as e:
                print(f"⚠️ Failed to add additional context: {e}")
                import traceback
                traceback.print_exc()
        
        # ========================================
        # 最終的な質問文の構築
        # ========================================
        # システムコンテキスト + 追加コンテキスト + ユーザー質問
        enhanced_question = context_prefix + question
        
        # ========================================
        # 点群のサンプリング処理
        # ========================================
        if len(point_cloud) > num_points:
            indices = np.random.choice(len(point_cloud), num_points, replace=False)
            point_cloud = point_cloud[indices]
        elif len(point_cloud) < num_points:
            indices = np.random.choice(len(point_cloud), num_points, replace=True)
            point_cloud = point_cloud[indices]
        
        # 色情報の追加と次元調整
        if point_cloud.shape[1] == 3:
            colors = np.ones((point_cloud.shape[0], 3)) * 0.8
            point_cloud = np.concatenate([point_cloud, colors], axis=1)
        point_cloud = point_cloud[:, :6]
        
        print(f"📊 Point cloud shape: {point_cloud.shape}")
        if len(enhanced_question) > 300:
            print(f"❓ Question: {enhanced_question[:500]}...[CONTEXT]...{enhanced_question[-100:]}")
        else:
            print(f"❓ Question: {enhanced_question[-100:]}")
        
        # Tensor変換
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud_tensor = torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0)
        else:
            point_cloud_tensor = point_cloud.unsqueeze(0) if point_cloud.dim() == 2 else point_cloud
        
        model_dtype = getattr(model, 'dtype', torch.float16)

        # 推論実行
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            point_cloud=point_cloud_tensor,
            question=enhanced_question,
            device=device,
            dtype=model_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        print(f"💡 Response: {response[:200]}...")
        return response
    
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg)

    
def batch_generate_responses(
    model,
    tokenizer,
    point_clouds: list,
    questions: list,
    device: torch.device,
    **kwargs
) -> list:
    """
    バッチ推論用
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_clouds: 点群データのリスト
        questions: 質問文のリスト
        device: デバイス
        **kwargs: generate_response_with_pointllmに渡す追加引数
    
    Returns:
        list: 応答のリスト
    """
    if len(point_clouds) != len(questions):
        raise ValueError("Number of point clouds and questions must match")
    
    responses = []
    for i, (pc, question) in enumerate(zip(point_clouds, questions)):
        print(f"\nProcessing {i+1}/{len(point_clouds)}")
        response = generate_response_with_pointllm(
            model=model,
            tokenizer=tokenizer,
            point_cloud=pc,
            question=question,
            device=device,
            **kwargs
        )
        responses.append(response)
    return responses


def generate_with_retry(
    model,
    tokenizer,
    point_cloud: np.ndarray,
    question: str,
    device: torch.device,
    max_retries: int = 3,
    **kwargs
) -> Optional[str]:
    """
    リトライ機能付きの推論実行
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群データ
        question: 質問文
        device: デバイス
        max_retries: 最大リトライ回数
        **kwargs: 追加の生成パラメータ
    
    Returns:
        Optional[str]: 生成された回答、失敗時はNone
    """
    for attempt in range(max_retries):
        try:
            response = generate_response_with_pointllm(
                model=model,
                tokenizer=tokenizer,
                point_cloud=point_cloud,
                question=question,
                device=device,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"❌ All {max_retries} attempts failed")
                return None
            print("Retrying...")
    
    return None


def validate_point_cloud(point_cloud: np.ndarray) -> bool:
    """
    点群データの妥当性をチェック
    
    Args:
        point_cloud: 点群データ
    
    Returns:
        bool: 妥当な場合True
    """
    if not isinstance(point_cloud, np.ndarray):
        print("❌ Point cloud must be a numpy array")
        return False
    
    if point_cloud.ndim != 2:
        print(f"❌ Point cloud must be 2D, got {point_cloud.ndim}D")
        return False
    
    # チャンネル数チェックを緩和 (3, 6, 7, 8 を許可)
    # ※ただし推論時は最初の6チャンネルのみ使われます
    if point_cloud.shape[1] not in [3, 6, 7, 8]:
        print(f"❌ Point cloud must have 3, 6, 7 or 8 channels, got {point_cloud.shape[1]}")
        return False
    
    if len(point_cloud) == 0:
        print("❌ Point cloud is empty")
        return False
    
    if np.isnan(point_cloud).any():
        print("❌ Point cloud contains NaN values")
        return False
    
    if np.isinf(point_cloud).any():
        print("❌ Point cloud contains Inf values")
        return False
    
    print(f"✅ Point cloud validation passed: {point_cloud.shape}")
    return True