import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    res = 0.
    unique_vals = np.unique(y, return_counts=True)
    for i in range(len(unique_vals[0])):
        probability = unique_vals[1][i]/len(y) # count divided by total amount
        res += -probability * (np.log2(probability))
    return res
    
    #if len(unique_vals[0]) == 1 and len(unique_vals[1]) == len(y):
    #    return 0.
    
        