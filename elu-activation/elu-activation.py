import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [z if z>0 else alpha *(math.exp(z)-1) for z in x]
    # Write code here