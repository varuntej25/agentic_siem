"""
GPU Acceleration Utilities
==========================

Lightweight utilities for automatic CUDA/GPU acceleration with safe CPU fallbacks.
Supports mixed precision, device management, and HuggingFace model optimization.

Environment Variables:
- USE_GPU: Set to 'false' to disable GPU usage (default: true)
- CUDA_DEVICE: Specify CUDA device ID (default: 0)
- AMP: Mixed precision mode: auto|fp16|bf16|off (default: auto)
- GPU_MEMORY_FRACTION: Limit GPU memory usage (0.0-1.0)
"""

from __future__ import annotations
import os
import sys
import torch
import logging
from typing import Optional, Union, Dict, Any, List
from contextlib import contextmanager, nullcontext
import warnings

def _supports_utf8_stream(stream) -> bool:
    """Check if a stream supports UTF-8 encoding"""
    try:
        enc = getattr(stream, "encoding", None)
        return bool(enc) and enc.lower().replace("-", "") == "utf8"
    except Exception:
        return False

def emoji(s: str) -> str:
    """Emit emoji only if stdout truly supports UTF-8; else empty string."""
    return s if _supports_utf8_stream(sys.stdout) else ""

# ----- policy toggles via env -----
def want_gpu() -> bool:
    """Default TRUE – force GPU if available"""
    return os.getenv("USE_GPU", "true").lower() in ("1","true","yes","y")

def amp_mode() -> str:
    """auto | fp16 | bf16 | off (default auto)"""
    return os.getenv("AMP", "auto").lower()

# ----- device selection -----
def get_device() -> torch.device:
    """Get optimal device based on availability and settings"""
    if want_gpu() and torch.cuda.is_available():
        device_id = int(os.getenv('CUDA_DEVICE', '0'))
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def device_info() -> dict:
    """Get comprehensive device information"""
    d = get_device()
    info = {"device": str(d), "device_type": d.type, "amp_enabled": False}
    if d.type == "cuda":
        info["name"] = torch.cuda.get_device_name(d)
        info["capability"] = torch.cuda.get_device_capability(d)
        info["memory_gb"] = torch.cuda.get_device_properties(d).total_memory / 1024**3
    return info

# ----- runtime perf knobs (safe) -----
def setup_runtime():
    """Setup runtime optimizations"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

# ----- AMP context -----
def amp_autocast():
    """Get appropriate AMP autocast context"""
    d = get_device()
    m = amp_mode()
    if d.type != "cuda" or m == "off":
        return nullcontext()
    if m == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if m == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    # auto preference: bf16 if possible, else fp16
    try:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    except Exception:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    

# ----- tensor & batch movement -----
def move_to_device(x, device=None):
    """Move tensors/dicts/lists to device recursively"""
    if device is None:
        device = get_device()
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        ys = [move_to_device(v, device) for v in x]
        return type(x)(ys) if isinstance(x, tuple) else ys
    return x

# ----- model placement -----
def to_cuda(model: torch.nn.Module):
    """Place model on GPU if available else CPU, return (model, device)."""
    device = get_device()
    model = model.to(device)
    return model, device

# ----- training helpers -----
def maybe_grad_scaler():
    """Get gradient scaler if using CUDA, else None"""
    return torch.cuda.amp.GradScaler() if get_device().type == "cuda" else None

# ----- Backward compatibility with existing GPUAccelerator class -----
class GPUAccelerator:
    """Legacy GPU accelerator class for backward compatibility"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUAccelerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.device = get_device()
            self.device_type = self.device.type
            self.amp_enabled = amp_mode() != "off" and self.device_type == "cuda"
            self._log_device_info()
            setup_runtime()
            GPUAccelerator._initialized = True
    
    def _log_device_info(self):
        """Log device and capability information"""
        if self.device_type == 'cuda':
            info = device_info()
            gpu_name = info.get("name", "Unknown GPU")
            memory_gb = info.get("memory_gb", 0)
            self.logger.info(f"GPU Acceleration Enabled: {gpu_name} ({memory_gb:.1f}GB)")
            if self.amp_enabled:
                self.logger.info("Mixed Precision (AMP) Enabled")
        else:
            self.logger.info("Using CPU for computation")
    
    @property
    def current_device(self) -> torch.device:
        """Get current device"""
        return self.device
    
    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA"""
        return self.device_type == 'cuda'
    
    def to_device(self, data: Union[torch.Tensor, Dict, List, Any], non_blocking: bool = False) -> Any:
        """Move data to appropriate device"""
        return move_to_device(data, self.device)
    
    def autocast_context(self):
        """Automatic Mixed Precision context manager"""
        return amp_autocast()
    
    def optimize_model(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """Optimize model for GPU"""
        model, _ = to_cuda(model)
        return model
    
    def optimize_transformers_model(self, model: Any, tokenizer: Any = None) -> tuple:
        """Optimize HuggingFace transformers models"""
        try:
            model, _ = to_cuda(model)
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Transformers optimization failed: {e}")
            return model, tokenizer
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Get optimal batch size based on available memory"""
        if self.device_type == 'cpu':
            return min(base_batch_size, 16)  # Conservative for CPU
        
        try:
            info = device_info()
            memory_gb = info.get("memory_gb", 8)
            if memory_gb >= 16:
                return base_batch_size * 2
            elif memory_gb >= 8:
                return base_batch_size
            else:
                return base_batch_size // 2
        except:
            return base_batch_size
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return device_info()


# Global accelerator instance for backward compatibility
_accelerator = None

def get_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance"""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator()
    return _accelerator

# Legacy function aliases for backward compatibility
def to_device(data: Any, non_blocking: bool = False) -> Any:
    """Move data to optimal device"""
    return move_to_device(data)

def autocast():
    """Get autocast context manager"""
    return amp_autocast()

def optimize_model(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Optimize model for current device"""
    model, _ = to_cuda(model)
    return model

def optimize_transformers_model(model: Any, tokenizer: Any = None) -> tuple:
    """Optimize HuggingFace model for current device"""
    try:
        model, _ = to_cuda(model)
        return model, tokenizer
    except Exception:
        return model, tokenizer

def clear_cache():
    """Clear GPU cache"""
    if get_device().type == "cuda":
        torch.cuda.empty_cache()

def get_optimal_batch_size(base_batch_size: int = 32) -> int:
    """Get optimal batch size for current device"""
    return get_accelerator().get_optimal_batch_size(base_batch_size)

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    return device_info()

# Utility decorators
def gpu_accelerated(func):
    """Decorator to automatically move function inputs to GPU"""
    def wrapper(*args, **kwargs):
        # Move tensor arguments to device
        args = tuple(move_to_device(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {k: move_to_device(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        
        with amp_autocast():
            return func(*args, **kwargs)
    return wrapper

def safe_emoji(s: str) -> str:
    """Emit emoji only if stdout truly supports UTF-8; else empty string."""
    return s if _supports_utf8_stream(sys.stdout) else ""

class GPUAccelerator:
    """Centralized GPU acceleration and device management"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUAccelerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._setup_device()
            self._setup_amp()
            self._log_device_info()
            GPUAccelerator._initialized = True
    
    def _setup_device(self):
        """Setup optimal device with environment variable overrides and clear fallback logging"""
        # Check for forced CPU usage
        if os.getenv('FORCE_CPU', '0') == '1':
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            wrench_emoji = emoji("🔧")
            self.logger.info(f"{wrench_emoji} Forced CPU usage via FORCE_CPU environment variable")
            return
        
        # Check CUDA availability with detailed logging
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            cpu_emoji = emoji("💻")
            self.logger.info(f"{cpu_emoji} CUDA not available (torch.cuda.is_available() = False), using CPU")
            return
        
        # CUDA is available, try to set it up
        try:
            # Get device ID from environment or default to 0
            device_id = int(os.getenv('CUDA_DEVICE', '0'))
            if device_id < torch.cuda.device_count():
                self.device = torch.device(f'cuda:{device_id}')
                self.device_type = 'cuda'
                
                # Set memory fraction if specified
                memory_fraction = os.getenv('GPU_MEMORY_FRACTION')
                if memory_fraction:
                    try:
                        fraction = float(memory_fraction)
                        if 0.0 < fraction <= 1.0:
                            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
                            wrench_emoji = emoji("🔧")
                            self.logger.info(f"{wrench_emoji} GPU memory limited to {fraction*100:.1f}%")
                    except ValueError:
                        self.logger.warning(f"Invalid GPU_MEMORY_FRACTION: {memory_fraction}")
                
            else:
                self.logger.warning(f"CUDA device {device_id} not available, using device 0")
                self.device = torch.device('cuda:0')
                self.device_type = 'cuda'
        except Exception as e:
            self.logger.warning(f"CUDA setup failed: {e}, falling back to CPU")
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
    
    def _setup_amp(self):
        """Setup Automatic Mixed Precision with clear logging"""
        self.amp_enabled = (
            self.device_type == 'cuda' and 
            os.getenv('ENABLE_AMP', 'True').lower() in ['true', '1', 'yes'] and
            torch.cuda.is_available() and 
            torch.cuda.get_device_capability()[0] >= 7  # Tensor Core support
        )
        
        if self.amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler()
            lightning_emoji = emoji("⚡")
            self.logger.info(f"{lightning_emoji} Mixed Precision (AMP) Enabled")
        else:
            self.scaler = None
    
    def _log_device_info(self):
        """Log device and capability information with UTF-8 safe emojis"""
        if self.device_type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            rocket_emoji = emoji("🚀")
            self.logger.info(f"{rocket_emoji} GPU Acceleration Enabled: {gpu_name} ({memory_gb:.1f}GB)")
            if self.amp_enabled:
                lightning_emoji = emoji("⚡")
                self.logger.info(f"{lightning_emoji} Mixed Precision (AMP) Enabled")
        else:
            cpu_emoji = emoji("💻")
            self.logger.info(f"{cpu_emoji} Using CPU for computation")
    
    @property
    def current_device(self) -> torch.device:
        """Get current device"""
        return self.device
    
    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA"""
        return self.device_type == 'cuda'
    
    def to_device(self, data: Union[torch.Tensor, Dict, List, Any], non_blocking: bool = False) -> Any:
        """Move data to appropriate device with smart handling"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=non_blocking)
        elif isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item, non_blocking) for item in data)
        else:
            return data
    
    @contextmanager
    def autocast_context(self):
        """Automatic Mixed Precision context manager"""
        if self.amp_enabled:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def optimize_model(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """Optimize model for GPU with best practices"""
        try:
            # Move model to device
            model = model.to(self.device)
            
            # Enable eval mode optimizations
            if not model.training:
                model.eval()
                
            # Compile model for PyTorch 2.0+ if available
            if hasattr(torch, 'compile') and self.is_cuda:
                try:
                    model = torch.compile(model, **kwargs)
                    fire_emoji = safe_emoji("🔥")
                    self.logger.info(f"{fire_emoji} Model compilation enabled (PyTorch 2.0+)")
                except Exception as e:
                    self.logger.debug(f"Model compilation skipped: {e}")
            
            return model
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model
    
    def optimize_transformers_model(self, model: Any, tokenizer: Any = None) -> tuple:
        """Optimize HuggingFace transformers models"""
        try:
            # Move model to device
            model = model.to(self.device)
            
            # Enable better memory usage
            if hasattr(model, 'gradient_checkpointing_enable') and self.is_cuda:
                try:
                    model.gradient_checkpointing_enable()
                except:
                    pass  # Not all models support this
            
            # Optimize attention if available
            if hasattr(model.config, 'torch_dtype') and self.is_cuda:
                if model.config.torch_dtype != torch.float16:
                    try:
                        model = model.half()  # Convert to fp16 for inference
                        cycle_emoji = safe_emoji("🔄")
                        self.logger.info(f"{cycle_emoji} Model converted to FP16 for faster inference")
                    except:
                        pass  # Some models don't support fp16
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Transformers optimization failed: {e}")
            return model, tokenizer
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Get optimal batch size based on available memory"""
        if self.device_type == 'cpu':
            return min(base_batch_size, 16)  # Conservative for CPU
        
        try:
            # Get available GPU memory
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            
            # Heuristic: larger memory = larger batch size
            if memory_gb >= 16:
                return base_batch_size * 2
            elif memory_gb >= 8:
                return base_batch_size
            else:
                return base_batch_size // 2
        except:
            return base_batch_size
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            'device': str(self.device),
            'device_type': self.device_type,
            'amp_enabled': self.amp_enabled,
        }
        
        if self.is_cuda:
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'memory_total_gb': torch.cuda.get_device_properties(self.device).total_memory / 1024**3,
                'memory_allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'memory_cached_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
            })
        
        return info


# Global accelerator instance
_accelerator = None

def get_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance"""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator()
    return _accelerator

def get_device() -> torch.device:
    """Get optimal device"""
    return get_accelerator().current_device

def to_device(data: Any, non_blocking: bool = False) -> Any:
    """Move data to optimal device"""
    return get_accelerator().to_device(data, non_blocking)

def autocast():
    """Get autocast context manager"""
    return get_accelerator().autocast_context()

def optimize_model(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Optimize model for current device"""
    return get_accelerator().optimize_model(model, **kwargs)

def optimize_transformers_model(model: Any, tokenizer: Any = None) -> tuple:
    """Optimize HuggingFace model for current device"""
    return get_accelerator().optimize_transformers_model(model, tokenizer)

def clear_cache():
    """Clear GPU cache"""
    get_accelerator().clear_cache()

def get_optimal_batch_size(base_batch_size: int = 32) -> int:
    """Get optimal batch size for current device"""
    return get_accelerator().get_optimal_batch_size(base_batch_size)

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    return get_accelerator().get_device_info()

# Utility decorators
def gpu_accelerated(func):
    """Decorator to automatically move function inputs to GPU"""
    def wrapper(*args, **kwargs):
        accelerator = get_accelerator()
        # Move tensor arguments to device
        args = tuple(accelerator.to_device(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {k: accelerator.to_device(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        
        with accelerator.autocast_context():
            return func(*args, **kwargs)
    return wrapper
