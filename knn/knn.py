from typing import List, Tuple, Any, Optional

import numpy as np

from class_models.movie_feature_type import MovieFeatureType


class KNN:
    def __init__(self, feature_types: List[Tuple[str, MovieFeatureType]], k: int = 5,
                 similarity_function: Any = None) -> None:
        self.k: int = k
        self.similarity_function: Any = similarity_function
        self.features: Optional[List[np.ndarray]] = None
        self.labels: Optional[List[int]] = None
        self.feature_types: List[Tuple[str, MovieFeatureType]] = feature_types

    def fit(self, features: List[np.ndarray], labels: List[int]) -> None:
        self.features = features
        self.labels = labels

    def predict(self, task_vector: np.ndarray) -> int:
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with training data.")

        similarities: List[Tuple[float, int]] = []
        for i in range(len(self.features)):
            feature_vector = self.features[i]
            similarity = self.similarity_function(task_vector, feature_vector, self.feature_types)
            similarities.append((similarity, i))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k: List[Tuple[float, int]] = similarities[:self.k]

        numerator: float = 0.0
        denominator: float = 0.0
        for similarity, i in top_k:
            label = self.labels[i]
            numerator += similarity * label
            denominator += similarity

        if denominator == 0:
            return int(round(np.mean(self.labels)))

        predicted_label = numerator / denominator
        return np.clip(int(round(predicted_label)), 0, 5)