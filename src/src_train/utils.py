"""
Utility functions.
"""

import gc
import torch


def clean_memory():
    """Free up GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
