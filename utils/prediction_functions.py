import csv
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Hashable
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from class_models.movie_features import MovieFeatures
from knn.knn import KNN

def predict_ratings(train_data_df: DataFrame, task_data_df: DataFrame,
                    movie_feature_vectors: Dict[int, np.ndarray], k_values: List[int],
                    report_filename: str = "knn_report.csv") -> dict[Hashable, int | Any]:
    user_ratings: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for _, row in train_data_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        rating = row['rating']
        user_ratings[user_id].append((movie_id, rating))

    predictions: dict[Hashable, int | Any] = {}
    best_k_counts: Dict[int, int] = defaultdict(int)
    k_scores: List[Tuple[int, float]] = []

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
                if len(k_values) == 1:
                    best_k = k_values[0]
                else:
                    with ThreadPoolExecutor() as executor:
                        k_scores = list(executor.map(lambda k: (k, cross_validate_k(X_train, y_train, k)), k_values))

                    best_k = max(k_scores, key=lambda x: x[1])[0]
                    best_k_counts[best_k] += 1

                knn = KNN(feature_types=MovieFeatures.feature_types(), k=best_k)
                knn.fit(X_train, y_train)
                predicted_rating = knn.predict(task_vector)
                predictions[i] = predicted_rating
            else:
                predictions[i] = int(round(np.mean(train_data_df['rating'])))
        else:
            predictions[i] = int(round(np.mean(train_data_df['rating'])))

    if len(k_values) > 1:
        save_k_report(best_k_counts, k_scores, report_filename)

    return predictions

def cross_validate_k(X_train: List[np.ndarray], y_train: List[int], k: int) -> float:
    knn = KNN(feature_types=MovieFeatures.feature_types(), k=k)
    return knn.cross_validate(X_train, y_train)

def save_k_report(best_k_counts: Dict[int, int], k_scores: List[Tuple[int, float]], report_filename: str) -> None:
    with open(report_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["k", "Count Chosen", "Average Accuracy"])

        k_accuracy_dict = {k: score for k, score in k_scores}

        for k, count in sorted(best_k_counts.items()):
            writer.writerow([k, count, k_accuracy_dict.get(k, "N/A")])

    print(f"K value report saved to {report_filename}")
