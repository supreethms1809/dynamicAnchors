"""
Device management utilities for consistent device handling across the codebase.
"""
import torch
from typing import Union


def _is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    MPS is Apple's GPU acceleration framework for PyTorch on macOS.
    
    Returns:
        True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _get_auto_device() -> torch.device:
    """
    Auto-detect the best available device.
    
    Priority order:
    1. CUDA (if available)
    2. MPS (if available, macOS Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif _is_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Standardize device handling: convert any device input to torch.device.
    
    Args:
        device: Device specification. Can be:
            - None: Auto-detect (cuda > mps > cpu)
            - str: "cpu", "cuda", "cuda:0", "mps", "auto", etc.
            - torch.device: Already a device object
    
    Returns:
        torch.device object
    
    Examples:
        >>> get_device()  # Auto-detect
        device(type='cpu')  # or 'cuda' or 'mps' depending on availability
        >>> get_device("cuda")
        device(type='cuda')
        >>> get_device("mps")
        device(type='mps')
        >>> get_device("auto")
        device(type='cpu')  # or 'cuda' or 'mps' if available
    """
    if device is None:
        # Auto-detect: use best available device
        device = _get_auto_device()
    elif isinstance(device, torch.device):
        # Already a device object, use as-is
        pass
    elif isinstance(device, str):
        if device.lower() == "auto":
            # Auto-detect
            device = _get_auto_device()
        else:
            # Convert string to device
            # Validate MPS availability if requested
            if device.lower() == "mps" and not _is_mps_available():
                raise RuntimeError(
                    "MPS (Metal Performance Shaders) is not available. "
                    "MPS requires macOS with Apple Silicon (M1/M2/M3). "
                    "Falling back to CPU is not automatic - please specify 'cpu' or 'auto'."
                )
            device = torch.device(device)
    else:
        raise ValueError(f"Invalid device type: {type(device)}. Expected str, torch.device, or None.")
    
    return device


def get_device_str(device: Union[str, torch.device, None] = None) -> str:
    """
    Get device as string representation.
    
    Args:
        device: Device specification (same as get_device)
    
    Returns:
        String representation of device (e.g., "cpu", "cuda", "cuda:0")
    """
    device_obj = get_device(device)
    return str(device_obj)


def get_device_pair(device: Union[str, torch.device, None] = None) -> tuple[torch.device, str]:
    """
    Get both device object and string representation.
    
    Useful when you need both formats (e.g., for PyTorch operations and string parameters).
    
    Args:
        device: Device specification (same as get_device)
    
    Returns:
        Tuple of (device_obj, device_str)
    
    Examples:
        >>> device_obj, device_str = get_device_pair("cuda")
        >>> device_obj
        device(type='cuda')
        >>> device_str
        'cuda'
    """
    device_obj = get_device(device)
    device_str = str(device_obj)
    return device_obj, device_str

