"""
Device Management Utility for PointLLM
======================================

This module provides device-agnostic utilities for PyTorch operations,
supporting CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon).

The module automatically detects available devices and provides a consistent
interface for tensor and model device placement across different hardware.

Usage:
    from pointllm.device_utils import get_device, move_to_device

    # Get the best available device
    device = get_device()

    # Move tensor to device
    tensor = move_to_device(tensor, device)

    # Move model to device
    model = move_to_device(model, device)

Author: Refactored for CPU/MPS compatibility
Date: 2025
"""

import torch
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> torch.device:
    """
    Get the appropriate PyTorch device for computation.

    This function automatically selects the best available device in the following priority:
    1. User-specified device (if provided)
    2. CUDA (NVIDIA GPU) if available
    3. MPS (Apple Silicon) if available
    4. CPU (fallback)

    Args:
        device (str or torch.device, optional): Specific device to use.
            Can be 'cpu', 'cuda', 'mps', 'cuda:0', etc.
            If None, automatically selects the best available device.
        verbose (bool): If True, prints device information. Default: False.

    Returns:
        torch.device: The selected PyTorch device.

    Examples:
        >>> device = get_device()  # Auto-select best device
        >>> device = get_device('cpu')  # Force CPU
        >>> device = get_device('cuda:0')  # Specific CUDA device

    Notes:
        - MPS backend requires PyTorch >= 1.12 and macOS >= 12.3
        - Some operations may not be fully supported on MPS and will
          automatically fall back to CPU when needed
    """
    # If user specifies a device, validate and return it
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise TypeError(f"device must be str or torch.device, got {type(device)}")

        if verbose:
            logger.info(f"Using user-specified device: {device}")
        return device

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Using GPU: {cuda_device_name}")
            logger.info(f"Number of CUDA devices: {cuda_device_count}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            logger.info("MPS (Apple Silicon) is available. Using MPS device.")
            logger.warning("Note: Some operations may not be fully supported on MPS and will fall back to CPU.")
    else:
        device = torch.device('cpu')
        if verbose:
            logger.info("No GPU available. Using CPU.")
            logger.warning("Performance may be significantly slower on CPU.")

    return device


def move_to_device(obj: Union[torch.Tensor, torch.nn.Module],
                   device: Optional[Union[str, torch.device]] = None,
                   non_blocking: bool = False) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move a PyTorch tensor or model to the specified device.

    This function provides a unified interface for moving objects to different devices,
    with automatic device selection if not specified.

    Args:
        obj (torch.Tensor or torch.nn.Module): The tensor or model to move.
        device (str or torch.device, optional): Target device. If None, uses get_device().
        non_blocking (bool): If True, tries to convert asynchronously with respect to
            the host if possible. Default: False.

    Returns:
        torch.Tensor or torch.nn.Module: The object moved to the target device.

    Examples:
        >>> tensor = torch.randn(3, 3)
        >>> tensor = move_to_device(tensor, 'cuda')

        >>> model = MyModel()
        >>> model = move_to_device(model, 'mps')

    Notes:
        - For MPS backend, some dtypes may not be supported and will automatically
          fall back to CPU
        - Use non_blocking=True for potential performance improvements with pinned memory
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        return obj.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        # Handle MPS-specific errors by falling back to CPU
        if 'mps' in str(device).lower() and ('not available' in str(e).lower() or 'not supported' in str(e).lower()):
            logger.warning(f"MPS operation failed: {e}. Falling back to CPU for this operation.")
            return obj.to('cpu', non_blocking=non_blocking)
        else:
            raise


def get_device_safe(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Safely get a device, with fallback logic for unsupported operations.

    This function attempts to use the preferred device, but falls back to CPU
    if the device is not available or supported.

    Args:
        preferred_device (str or torch.device, optional): Preferred device to use.

    Returns:
        torch.device: A safe, available device.
    """
    try:
        device = get_device(preferred_device)
        # Test if device is actually usable
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        return device
    except (RuntimeError, AssertionError) as e:
        logger.warning(f"Device {preferred_device} not available: {e}. Falling back to CPU.")
        return torch.device('cpu')


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Dictionary containing device information including:
            - cuda_available (bool)
            - cuda_device_count (int)
            - cuda_device_names (list)
            - mps_available (bool)
            - current_device (torch.device)
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_device_names': [],
        'mps_available': is_mps_available(),
        'current_device': get_device(),
    }

    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_names'] = [torch.cuda.get_device_name(i)
                                      for i in range(info['cuda_device_count'])]

    return info


def print_device_info():
    """Print detailed device information to console."""
    info = get_device_info()
    print("\n" + "="*50)
    print("PyTorch Device Information")
    print("="*50)
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        for i, name in enumerate(info['cuda_device_names']):
            print(f"  Device {i}: {name}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"Current Device: {info['current_device']}")
    print("="*50 + "\n")


# Set random seed for reproducibility across all devices
def set_random_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility across CPU, CUDA, and MPS.

    Args:
        seed (int): Random seed value
        deterministic (bool): Whether to set deterministic mode for CUDNN.
            This may reduce performance but ensures reproducibility.
            Note: Only affects CUDA backend.

    Notes:
        - For CUDA: Sets torch.cuda.manual_seed_all()
        - For MPS: Uses torch.manual_seed() (MPS shares CPU seed)
        - Sets numpy and random seeds as well
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDNN deterministic mode enabled for reproducibility.")

    # MPS uses the same seed as CPU
    if is_mps_available():
        logger.info("Random seed set for MPS (shares CPU seed).")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print_device_info()

    # Test device selection
    device = get_device(verbose=True)
    print(f"\nSelected device: {device}")

    # Test tensor operations
    test_tensor = torch.randn(3, 3)
    print(f"\nOriginal tensor device: {test_tensor.device}")

    test_tensor = move_to_device(test_tensor, device)
    print(f"After move_to_device: {test_tensor.device}")

    # Test random seed
    set_random_seed(42)
    print("\nRandom seed set to 42")
