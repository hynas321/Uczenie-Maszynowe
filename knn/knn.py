import numpy as np

from utils.similarity_functions import compute_similarity

class KNN:
    def __init__(self, feature_types, k=5, similarity_function=compute_similarity):
        self.k = k
        self.similarity_function = similarity_function
        self.features = None
        self.labels = None
        self.feature_types = feature_types

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, task_vector):
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with training data.")

        similarities = []
        for i in range(len(self.features)):
            feature_vector = self.features[i]
            similarity = self.similarity_function(task_vector, feature_vector, self.feature_types)
            similarities.append((similarity, i))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k]

        numerator = 0.0
        denominator = 0.0
        for similarity, i in top_k:
            label = self.labels[i]
            numerator += similarity * label
            denominator += similarity

        if denominator == 0:
            return int(round(np.mean(self.labels)))

        predicted_label = numerator / denominator

        return np.clip(int(round(predicted_label)), 0, 5)