from typing import Union
import numpy as np
import torch


def to_int32(number: Union[int, np.ndarray, torch.Tensor, str]) -> Union[np.int32, str]:
    """Convert integers to 32-bit precision, leave strings unchanged.
    
    The size of our dstore (116988150) can be encoded in 32-bit precision.
    """
    # return number
    if isinstance(number, torch.Tensor):
        number = number.item()
        return np.int32(number)
    elif isinstance(number, int):
        return np.int32(number)
    elif isinstance(number, np.ndarray):
        return number.astype(np.int32)
    else:  # String or other: leave unchanged.
        return number