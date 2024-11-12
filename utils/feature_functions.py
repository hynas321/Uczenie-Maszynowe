import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
from class_models.movie_feature_type import MovieFeatureType


def create_feature_vectors(movie_features_dict, feature_types):
    movie_ids = list(movie_features_dict.keys())
    features = []

    genre_label_dict, label_encoder = _label_encode_genres(movie_features_dict)

    for movie_id in movie_ids:
        movie_feature = movie_features_dict[movie_id]
        feature_vector = []

        for feature_name, feature_type in feature_types:
            value = getattr(movie_feature, feature_name, None)

            if feature_type == MovieFeatureType.NUMERICAL:
                feature_vector.append(_process_numerical_feature(feature_name, value))
            elif feature_name == 'genres' and feature_type == MovieFeatureType.CATEGORICAL:
                feature_vector.append(genre_label_dict[movie_id])
            else:
                feature_vector.append(0)

        features.append(feature_vector)

    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform([f[:-1] for f in features])
    combined_features = np.column_stack((numerical_features, [f[-1] for f in features]))

    movie_feature_vectors = {movie_id: combined_features[i] for i, movie_id in enumerate(movie_ids)}

    return movie_feature_vectors

def _process_numerical_feature(feature_name, value):
    if feature_name == 'release_date':
        release_date = datetime.strptime(value, "%Y-%m-%d")
        return (release_date - datetime(1970, 1, 1)).days
    return value

def _label_encode_genres(movie_features_dict):
    genre_combinations = [';'.join(map(str, features.genres)) for features in movie_features_dict.values()]
    label_encoder = LabelEncoder()
    genre_labels = label_encoder.fit_transform(genre_combinations)
    genre_label_dict = {movie_id: label for movie_id, label in zip(movie_features_dict.keys(), genre_labels)}

    return genre_label_dict, label_encoder

