"""
Reproducibility utilities for consistent random seed management.

This module ensures reproducible results across different runs by properly
setting random seeds for all relevant libraries.
"""

import os
import random
import numpy as np


def set_random_seeds(seed=1234):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value. Default is 1234.
        
    Note:
        For complete reproducibility, PYTHONHASHSEED environment variable
        must be set before Python starts:
        - Linux/Mac: export PYTHONHASHSEED=0
        - Windows: set PYTHONHASHSEED=0
        
        TensorFlow determinism requires additional configuration:
        - TF 2.x: tf.config.experimental.enable_op_determinism()
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Enable deterministic operations in TF 2.x
        if hasattr(tf.config.experimental, 'enable_op_determinism'):
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
    
    # Verify PYTHONHASHSEED is set
    if os.environ.get('PYTHONHASHSEED') != '0':
        import warnings
        warnings.warn(
            "PYTHONHASHSEED is not set to 0. For complete reproducibility, "
            "set PYTHONHASHSEED=0 before starting Python.",
            UserWarning
        )
    
    print(f"Random seeds set to {seed} for reproducibility")
