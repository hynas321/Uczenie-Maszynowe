import os

from dotenv import load_dotenv

from collaborative_filtering_ph_version.algorithms.collaborative_filtering import CollaborativeFiltering
from collaborative_filtering_ph_version.services.tmdb_api_service import TmdbApiService
from collaborative_filtering_ph_version.utils.csv_functions import save_accuracies_to_csv, load_csv_data, \
    save_predictions_to_csv, load_or_fetch_movie_features
from collaborative_filtering_ph_version.utils.feature_functions import create_feature_vectors


def main():
    load_dotenv()
    api_key: str | None = os.getenv('TMDB_API_KEY')

    if api_key is None:
        raise TypeError("API key not found")

    movie_df, task_df, train_df = load_csv_data()

    tmdb_id_mapping = movie_df['tmdb_movie_id'].to_dict()
    tmdb_service_instance = TmdbApiService(api_key)
    fetched_movie_features = load_or_fetch_movie_features(tmdb_id_mapping, tmdb_service_instance)

    feature_vectors_mapping, feature_column_names = create_feature_vectors(fetched_movie_features)
    unique_users = train_df['user_id'].unique()

    learning_rates = [0.001]
    epoch_counts = [50]

    collaborative_filtering = CollaborativeFiltering(learning_rates, epoch_counts,
                                                  num_features=len(feature_column_names))

    prediction_df, user_accuracies = collaborative_filtering.execute(unique_users, train_df, task_df,
                                                        feature_vectors_mapping)

    save_accuracies_to_csv(user_accuracies, 'user_accuracies.csv')
    save_predictions_to_csv(prediction_df, 'submission.csv')


if __name__ == '__main__':
    main()
