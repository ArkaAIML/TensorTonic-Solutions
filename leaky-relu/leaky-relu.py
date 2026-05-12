import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    y=np.array(x)
    return np.maximum(alpha*y,y)
    pass