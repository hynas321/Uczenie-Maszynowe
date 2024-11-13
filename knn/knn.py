from typing import List, Tuple, Any, Optional
import numpy as np
from utils.similarity_functions import compute_similarity

class KNN:
    def __init__(self, feature_types: List[Tuple[str, Any]], k: int = 5,
                 similarity_function=compute_similarity) -> None:
        self.k: int = k
        self.similarity_function: Any = similarity_function
        self.features: Optional[List[np.ndarray]] = None
        self.labels: Optional[List[int]] = None
        self.feature_types: List[Tuple[str, Any]] = feature_types

    def fit(self, features: List[np.ndarray], labels: List[int]) -> None:
        self.features = features
        self.labels = labels

    def predict(self, task_vector: np.ndarray) -> int:
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with training data.")

        similarities: List[Tuple[float, int]] = [
            (self.similarity_function(task_vector, feature_vector, self.feature_types), i)
            for i, feature_vector in enumerate(self.features)
        ]

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k]

        numerator = sum(similarity * self.labels[i] for similarity, i in top_k)
        denominator = sum(similarity for similarity, _ in top_k)

        if denominator == 0:
            return int(round(np.mean(self.labels)))

        predicted_label = numerator / denominator
        return np.clip(int(round(predicted_label)), 0, 5)