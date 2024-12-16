from typing import List

import pandas as pd
from tqdm import tqdm

from collaborative_filtering_ph_version.algorithms.model_optimizer import ModelOptimizer


class CollaborativeFiltering:
    def __init__(self, learning_rates: List[float], num_epochs: List[int], num_features: int):
        self.optimizer = ModelOptimizer(learning_rates, num_epochs, num_features)

    def execute(self, user_ids, training_data, task_data, feature_vector_mapping):
        user_accuracy_mapping = {}
        prediction_records = []

        for idx, user_id in enumerate(tqdm(user_ids, desc="Processing Users", unit="user", ncols=80), start=1):
            user_training_data = training_data[training_data['user_id'] == user_id]
            user_task_data = task_data[task_data['user_id'] == user_id]

            training_pairs = []
            for _, row in user_training_data.iterrows():
                training_pairs.append((feature_vector_mapping[row.movie_id], row.rating))

            data_splits = []
            for i in range(5):
                data_splits.append(training_pairs[i::5])

            best_lr, best_epoch, best_acc = self.optimizer.find_optimal_hyperparameters(data_splits)

            feature_matrix = []
            actual_ratings = []
            for features, rating in training_pairs:
                feature_matrix.append(features)
                actual_ratings.append(rating)

            optimized_params = self.optimizer.optimize_user_params(feature_matrix, actual_ratings, best_lr, best_epoch)

            for _, row in user_task_data.iterrows():
                movie_id = row.movie_id
                feature_vector = feature_vector_mapping.get(movie_id)
                if feature_vector is not None:
                    estimated_rating = self.optimizer.calculate_prediction(optimized_params, feature_vector)
                    prediction_records.append({
                        'index': row.name,
                        'user_id': user_id,
                        'movie_id': movie_id,
                        'rating': estimated_rating
                    })

            user_accuracy_mapping[idx] = best_acc

        predictions_dataframe = pd.DataFrame(prediction_records)

        return predictions_dataframe, user_accuracy_mapping