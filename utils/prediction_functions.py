from collections import defaultdict
from typing import Dict, Tuple, List, Any, Hashable

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from class_models.movie_features import MovieFeatures
from knn.knn import KNN

def predict_ratings(train_data_df: DataFrame, task_data_df: DataFrame,
                    movie_feature_vectors: Dict[int, np.ndarray]) -> dict[Hashable, int | Any]:
    user_ratings: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for _, row in train_data_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        rating = row['rating']
        user_ratings[user_id].append((movie_id, rating))

    predictions: dict[Hashable, int | Any] = {}

    for i, row in tqdm(task_data_df.iterrows(), total=task_data_df.shape[0], desc="Predicting ratings"):
        user_id = row['user_id']
        movie_id = row['movie_id']

        if movie_id not in movie_feature_vectors:
            predictions[i] = 3
            continue

        task_vector = movie_feature_vectors[movie_id]

        if user_id in user_ratings:
            user_movies = user_ratings[user_id]
            X_train: List[np.ndarray] = []
            y_train: List[int] = []
            for rated_movie_id, rating in user_movies:
                if rated_movie_id in movie_feature_vectors:
                    X_train.append(movie_feature_vectors[rated_movie_id])
                    y_train.append(rating)

            if X_train:
                knn = KNN(feature_types=MovieFeatures.feature_types(), k=5)
                knn.fit(X_train, y_train)
                predicted_rating = knn.predict(task_vector)
                predictions[i] = predicted_rating
            else:
                predictions[i] = int(round(np.mean(train_data_df['rating'])))
        else:
            predictions[i] = int(round(np.mean(train_data_df['rating'])))

    return predictions