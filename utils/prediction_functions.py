from collections import defaultdict

import numpy as np
from tqdm import tqdm

from knn.knn import KNN
from utils.similarity_functions import cosine_similarity


def predict_ratings(train_data_df, task_data_df, movie_feature_vectors):
    user_ratings = defaultdict(list)
    for _, row in train_data_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        rating = row['rating']
        user_ratings[user_id].append((movie_id, rating))

    predictions = {}

    for idx, row in tqdm(task_data_df.iterrows(), total=task_data_df.shape[0], desc="Predicting ratings"):
        user_id = row['user_id']
        movie_id = row['movie_id']

        if movie_id not in movie_feature_vectors:
            predictions[idx] = 3
            continue

        task_vector = movie_feature_vectors[movie_id]

        if user_id in user_ratings:
            user_movies = user_ratings[user_id]
            X_train = []
            y_train = []
            for rated_movie_id, rating in user_movies:
                if rated_movie_id in movie_feature_vectors:
                    X_train.append(movie_feature_vectors[rated_movie_id])
                    y_train.append(rating)

            if X_train:
                knn = KNN(k=5, similarity_function=cosine_similarity)
                knn.fit(X_train, y_train)
                predicted_rating = knn.predict(task_vector)
                predictions[idx] = predicted_rating
            else:
                predictions[idx] = int(round(np.mean(train_data_df['rating'])))
        else:
            predictions[idx] = int(round(np.mean(train_data_df['rating'])))

    return predictions