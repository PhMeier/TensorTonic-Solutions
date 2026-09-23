import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    if len(x) != len(p): return False
    prod = 0.
    for i in range(len(x)):
        prod += x[i]*p[i]
    return prod