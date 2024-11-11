import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from datetime import datetime
from class_models.movie_feature_type import MovieFeatureType

def create_feature_vectors(movie_features_dict, feature_types):
    movie_ids = list(movie_features_dict.keys())
    features = []
    genre_list = []

    for movie_id in movie_ids:
        movie_feature = movie_features_dict[movie_id]
        feature_vector = []

        for feature_name, feature_type in feature_types:
            value = getattr(movie_feature, feature_name, None)

            if feature_type == MovieFeatureType.NUMERICAL:
                feature_vector.append(_process_numerical_feature(feature_name, value))
            elif feature_type == MovieFeatureType.CATEGORICAL:
                genre_list.append(value)
            else:
                feature_vector.append(0)

        features.append(feature_vector)

    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(features)

    genre_features, genre_id_to_index = _encode_genre_features(genre_list)
    combined_features = np.hstack((numerical_features, genre_features))

    movie_feature_vectors = {movie_id: combined_features[i] for i, movie_id in enumerate(movie_ids)}

    return movie_feature_vectors, genre_id_to_index

def _process_numerical_feature(feature_name, value):
    if feature_name == 'release_date':
        release_date = datetime.strptime(value, "%Y-%m-%d")
        return (release_date - datetime(1970, 1, 1)).days
    return value

def _encode_genre_features(genre_list):
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(genre_list)
    genre_id_to_index = {genre_id: idx for idx, genre_id in enumerate(mlb.classes_)}
    return genre_features, genre_id_to_index
