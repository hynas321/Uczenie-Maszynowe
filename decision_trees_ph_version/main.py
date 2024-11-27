import os
from typing import Dict

from dotenv import load_dotenv

from class_models.movie_features import MovieFeatures
from services.tmdb_api_service import TmdbApiService
from utils.csv_functions import load_csv_data, load_or_fetch_movie_features, save_predictions_to_csv
from utils.feature_functions import create_feature_vectors
from utils.prediction_functions import predict_ratings

def main():
    load_dotenv()
    api_key: str | None = os.getenv('TMDB_API_KEY')

    if api_key is None:
        raise TypeError("API key not found")

    movie_data_df, task_data_df, train_data_df = load_csv_data()

    tmdb_api_service = TmdbApiService(api_key)

    movie_id_tmdb_id_dict: Dict[int, int] = movie_data_df['tmdb_movie_id'].to_dict()
    movie_features_dict: Dict[int, MovieFeatures] = load_or_fetch_movie_features(movie_id_tmdb_id_dict, tmdb_api_service)
    movie_feature_vectors_dict, feature_names = create_feature_vectors(movie_features_dict)

    tree_hyperparams = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }

    forest_hyperparams = {
        'n_trees': [5, 10, 15, 20],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }

    predictions_tree = predict_ratings(train_data_df, task_data_df, movie_feature_vectors_dict,
                                       tree_hyperparams, 'tree', report_filename='tree_hyperparams_report.csv',
                                       feature_names=feature_names)
    save_predictions_to_csv(task_data_df, predictions_tree, train_data_df, 'submission_tree.csv')

    predictions_forest = predict_ratings(train_data_df, task_data_df, movie_feature_vectors_dict,
                                         forest_hyperparams, 'forest', report_filename='forest_hyperparams_report.csv',
                                         feature_names=feature_names)
    save_predictions_to_csv(task_data_df, predictions_forest, train_data_df, 'submission_forest.csv')

if __name__ == "__main__":
    main()
