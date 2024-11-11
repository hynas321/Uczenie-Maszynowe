import numpy as np

from class_models.movie_feature_type import MovieFeatureType

def numerical_similarity(a, b):
    return 1 - abs(a - b) / (abs(a) + abs(b) + 1e-5)

def categorical_similarity(a, b):
    return 1 if a == b else 0

def compute_similarity(a, b, feature_types):
    similarities = []
    for i, (feature_name, feature_type) in enumerate(feature_types):
        if feature_type == MovieFeatureType.NUMERICAL:
            similarity = numerical_similarity(a[i], b[i])
        elif feature_type == MovieFeatureType.CATEGORICAL:
            similarity = categorical_similarity(a[i], b[i])
        else:
            similarity = 0
        similarities.append(similarity)
    return sum(similarities) / len(similarities)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)