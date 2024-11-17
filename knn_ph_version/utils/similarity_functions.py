from typing import List, Tuple
import numpy as np
from class_models.movie_feature_type import MovieFeatureType

def compute_similarity(a: np.ndarray, b: np.ndarray, feature_types: List[Tuple[str, MovieFeatureType]]) -> float:
    """
    Computes the similarity between two feature vectors based on feature types.
    """
    similarities = []
    index = 0

    for feature_name, feature_type in feature_types:
        if feature_type == MovieFeatureType.CATEGORICAL:
            num_categories = len(a) - index
            category_a = a[index:index + num_categories]
            category_b = b[index:index + num_categories]
            similarity = categorical_similarity(category_a, category_b)
            similarities.append(similarity)
            index += num_categories
        elif feature_type == MovieFeatureType.NUMERICAL:
            similarity = numerical_similarity(a[index], b[index])
            similarities.append(similarity)
            index += 1
        else:
            similarities.append(0)
            index += 1

    return sum(similarities) / len(similarities)

def categorical_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Jaccard similarity between two categorical vectors.
    """
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return intersection / union if union != 0 else 0

def numerical_similarity(a: float, b: float, max_range: float = 1.0) -> float:
    """
    Computes the similarity between two numerical values using Euclidean distance.
    """
    distance = abs(a - b)
    similarity = 1 - (distance / max_range)
    return max(0, min(1, similarity))
