import os
from typing import Dict, Hashable, Any

import numpy as np
from dotenv import load_dotenv

from class_models.movie_features import MovieFeatures
from services.tmdb_api_service import TmdbApiService
from utils.csv_functions import load_csv_data, load_or_fetch_movie_features, save_predictions_to_csv
from utils.feature_functions import create_feature_vectors
from utils.prediction_functions import predict_ratings

def main() -> None:
    load_dotenv()
    api_key: str | None = os.getenv('TMDB_API_KEY')

    movie_data_df, task_data_df, train_data_df = load_csv_data()

    tmdb_api_service: TmdbApiService = TmdbApiService(api_key)
    movie_id_tmdb_id_dict: Dict[int, int] = movie_data_df['tmdb_movie_id'].to_dict()

    movie_features_dict: Dict[int, MovieFeatures] = load_or_fetch_movie_features(movie_id_tmdb_id_dict, tmdb_api_service)
    movie_feature_vectors_dict: Dict[int, np.ndarray] = create_feature_vectors(movie_features_dict, MovieFeatures.feature_types())

    predictions_dict: dict[Hashable, int | Any] = predict_ratings(train_data_df, task_data_df, movie_feature_vectors_dict)
    save_predictions_to_csv(task_data_df, predictions_dict, train_data_df)

if __name__ == "__main__":
    main()
