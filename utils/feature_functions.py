from typing import List, Tuple, Dict, Union, Optional

import numpy as np
from datetime import datetime
from class_models.movie_feature_type import MovieFeatureType
from class_models.movie_features import MovieFeatures
from knn.label_encoder import LabelEncoder
from knn.min_max_scaler import MinMaxScaler


def create_feature_vectors(movie_features_dict: Dict[int, MovieFeatures],
                           feature_types: List[Tuple[str, MovieFeatureType]]) -> Dict[int, np.ndarray]:
    movie_ids: List[int] = list(movie_features_dict.keys())
    features: List[List[Union[int, float]]] = []

    genre_label_dict, label_encoder = _label_encode_genres(movie_features_dict)

    for movie_id in movie_ids:
        movie_feature = movie_features_dict[movie_id]
        feature_vector: List[Union[int, float]] = []

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
    numerical_features = scaler.fit_transform(np.array([f[:-1] for f in features]))
    combined_features = np.column_stack((numerical_features, [f[-1] for f in features]))

    movie_feature_vectors: Dict[int, np.ndarray] = {movie_id: combined_features[i] for i, movie_id in enumerate(movie_ids)}

    return movie_feature_vectors

def _process_numerical_feature(feature_name: str, value: Optional[str]) -> Union[int, float]:
    if feature_name == 'release_date' and value is not None:
        release_date = datetime.strptime(value, "%Y-%m-%d")
        return (release_date - datetime(1970, 1, 1)).days
    return value or 0

def _label_encode_genres(movie_features_dict: Dict[int, MovieFeatures]) -> Tuple[Dict[int, int], LabelEncoder]:
    genre_combinations = [';'.join(map(str, features.genres)) for features in movie_features_dict.values()]
    unique_genre_combinations = list(set(genre_combinations))

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_genre_combinations)

    genre_label_dict: Dict[int, int] = {
        movie_id: label_encoder.transform([';'.join(map(str, features.genres))])[0]
        for movie_id, features in movie_features_dict.items()
    }

    return genre_label_dict, label_encoder