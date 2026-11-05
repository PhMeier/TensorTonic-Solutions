def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    top_k_elements = recommended[:k]
    precision_at_k = float(len((set(top_k_elements).intersection(set(relevant)))) / k)
    recall_at_k = float(len((set(top_k_elements).intersection(set(relevant)))) / len(relevant))
    return [precision_at_k, recall_at_k]
    