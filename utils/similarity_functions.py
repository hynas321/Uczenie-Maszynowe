import numpy as np


def numerical_similarity(a, b):
    return 1 - abs(a - b) / (abs(a) + abs(b) + 1e-5)

def categorical_similarity(a, b):
    return 1 if a == b else 0

def set_similarity(a, b):
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union != 0 else 0

def compute_similarity(a, b, feature_types):
    similarities = []
    for i, (feature_name, feature_type) in enumerate(feature_types):
        if feature_type == 'numerical':
            sim = numerical_similarity(a[i], b[i])
        elif feature_type == 'categorical':
            sim = categorical_similarity(a[i], b[i])
        elif feature_type == 'set':
            sim = set_similarity(set(a[i]), set(b[i]))
        else:
            sim = 0
        similarities.append(sim)
    return sum(similarities) / len(similarities)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.sqrt(np.dot(v1, v1))
    norm_v2 = np.sqrt(np.dot(v2, v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    else:
        return dot_product / (norm_v1 * norm_v2)