from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from datetime import datetime

from collaborative_filtering_users_ph_version.algorithms.min_max_scaler import MinMaxScaler
from collaborative_filtering_users_ph_version.class_models.movie_features import MovieFeatures
from collaborative_filtering_users_ph_version.class_models.movie_features_type import MovieFeatureType


def create_feature_vectors(movie_features_dict: Dict[int, MovieFeatures]) -> Tuple[Dict[int, np.ndarray], List[str]]:
    all_features = []

    unique_genres: set[int] = {genre for movie in movie_features_dict.values() for genre in movie.genres}
    genre_encoder: dict = {genre: i for i, genre in enumerate(sorted(unique_genres))}

    genre_feature_names = [f'genre_{genre_id}' for genre_id in sorted(unique_genres)]

    movie_ids = list(movie_features_dict.keys())

    for movie_id in movie_ids:
        movie_feature = movie_features_dict[movie_id]
        feature_vector = _build_feature_vector(movie_feature, genre_encoder)
        all_features.append(feature_vector)

    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(np.array([f[:7] for f in all_features]))
    genre_features = np.array([f[7:] for f in all_features])
    combined_features = np.hstack((numerical_features, genre_features))

    movie_feature_vectors_dict = {movie_id: combined_features[i] for i, movie_id in enumerate(movie_ids)}

    numerical_feature_names = [name for name, feature_type in MovieFeatures.feature_types() if feature_type == MovieFeatureType.NUMERICAL]
    feature_names = numerical_feature_names + genre_feature_names

    return movie_feature_vectors_dict, feature_names

def _build_feature_vector(movie_features: MovieFeatures, genre_encoder: Dict[int, int]) -> List[Union[int, float]]:
    feature_vector = []
    feature_types: list[tuple[str, MovieFeatureType]] =  MovieFeatures.feature_types()

    for feature_name, feature_type in feature_types:
        value = getattr(movie_features, feature_name, None)

        if feature_type == MovieFeatureType.NUMERICAL:
            feature_vector.append(_process_numerical_feature(feature_name, value))
        elif feature_name == 'genres' and feature_type == MovieFeatureType.CATEGORICAL:
            genre_vector = [1 if genre in movie_features.genres else 0 for genre in genre_encoder]
            feature_vector.extend(genre_vector)
        else:
            feature_vector.append(0)

    return feature_vector

def _process_numerical_feature(feature_name: str, value: Optional[str]) -> Union[int, float]:
    if feature_name == 'release_date' and value:
        try:
            release_date = datetime.strptime(value, "%Y-%m-%d")
            return (release_date - datetime(1900, 1, 1)).days
        except ValueError:
            return 0
    return value if value is not None else 0