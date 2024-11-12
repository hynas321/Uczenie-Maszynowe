import os

from dotenv import load_dotenv

from class_models.movie_features import MovieFeatures
from services.tmdb_api_service import TmdbApiService
from utils.csv_functions import load_csv_data, load_or_fetch_movie_features, save_predictions_to_csv
from utils.feature_functions import create_feature_vectors
from utils.prediction_functions import predict_ratings

def main():
    load_dotenv()
    api_key = os.getenv('TMDB_API_KEY')

    movie_data_df, task_data_df, train_data_df = load_csv_data()
    tmdb_api_service = TmdbApiService(api_key)
    movie_id_tmdb_ids = movie_data_df['tmdb_movie_id'].to_dict()

    movie_features_dict = load_or_fetch_movie_features(movie_id_tmdb_ids, tmdb_api_service)
    movie_feature_vectors = create_feature_vectors(movie_features_dict, MovieFeatures.feature_types())
    predictions = predict_ratings(train_data_df, task_data_df, movie_feature_vectors)
    save_predictions_to_csv(task_data_df, predictions, train_data_df)

if __name__ == "__main__":
    main()