from collaborative_filtering_ph_version.utils.csv_functions import load_csv_data, save_predictions_to_csv, \
    save_accuracies_to_csv
from collaborative_filtering_ph_version.utils.similarity_functions import validate_and_process_ratings


def main():
    movie_data_df, task_data_df, train_data_df = load_csv_data()

    task_data_filled, user_accuracies = validate_and_process_ratings(train_data_df, task_data_df)

    save_predictions_to_csv(task_data_filled, 'submission.csv')
    save_accuracies_to_csv(user_accuracies, 'user_accuracy_report.csv')

if __name__ == "__main__":
    main()
