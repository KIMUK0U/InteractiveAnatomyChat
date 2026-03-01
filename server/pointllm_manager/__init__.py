"""
PointLLM Manager Package (LoRA対応版)

モデル管理と推論実行を提供
"""

from .model_manager import ModelManager
from .generate_response import (
    generate_response_with_pointllm,
    batch_generate_responses,
    generate_with_retry,
    validate_point_cloud
)
from .modules.head_space_lps_analyzer import (
    compute_lps_from_rotation,
    create_system_context_from_rotation,
    HeadSpaceLPSAnalyzer
)
__all__ = [
    'ModelManager',
    'generate_response_with_pointllm',
    'batch_generate_responses',
    'generate_with_retry',
    'validate_point_cloud',
    'compute_lps_from_rotation',
    'create_system_context_from_rotation',
    'HeadSpaceLPSAnalyzer'
]

__version__ = '1.0.0-lora'
