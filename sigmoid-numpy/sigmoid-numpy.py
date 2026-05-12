import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    y=np.array(x)
    act = 1 / (1 + np.exp(-y))
    return act