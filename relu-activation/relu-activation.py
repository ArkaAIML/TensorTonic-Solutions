import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    y=np.array(x)
    return np.maximum(0,x)
    pass