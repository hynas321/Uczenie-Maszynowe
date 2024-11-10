import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from datetime import datetime

def create_feature_vectors(movie_features_dict):
    movie_ids = list(movie_features_dict.keys())
    features = []
    genre_list = []

    for movie_id in movie_ids:
        movie_feature = movie_features_dict[movie_id]
        release_date = datetime.strptime(movie_feature.release_date, "%Y-%m-%d")
        days_since_reference = (release_date - datetime(1970, 1, 1)).days

        features.append([
            days_since_reference,
            movie_feature.runtime,
            movie_feature.budget,
            movie_feature.revenue,
            movie_feature.popularity
        ])
        genre_list.append(movie_feature.genres)

    features = np.array(features)

    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(features)

    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(genre_list)
    genre_id_to_index = {genre_id: idx for idx, genre_id in enumerate(mlb.classes_)}

    combined_features = np.hstack((numerical_features, genre_features))
    movie_feature_vectors = {movie_id: combined_features[idx] for idx, movie_id in enumerate(movie_ids)}

    return movie_feature_vectors, genre_id_to_index
