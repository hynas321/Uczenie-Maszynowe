import time
from collections import Counter

from rg_version.utils.data_fetch import get_users
from rg_version.utils.similiarity_functions import get_movies_similarity


def main() -> None:
    start_time = time.time()

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    users = get_users()
    for user in users[:1]:
        train_data = user.train_movies

        data_slices = divide_list(train_data)

        for k in k_values[14:]:

            for i, test_slice in enumerate(data_slices):

                test_set = test_slice
                train_set = [item for j, slice in enumerate(data_slices) if j != i for item in slice]
                accuracy = 0

                for test_point in test_set:

                    points = []

                    for train_point in train_set:
                        score = get_movies_similarity(test_point, train_point)
                        points.append([score, train_point.rating])

                    sorted_points = sorted(points, key=lambda x: x[0])

                    k_points = sorted_points[:k]

                    # Extract the ratings (second element of each sublist)
                    ratings = [point[1] for point in k_points]

                    frequency = Counter(ratings)

                    most_common = frequency.most_common()  # Returns list of (rating, count) tuples
                    max_count = most_common[0][1]  # Maximum occurrence count

                    # Collect all ratings with the highest frequency
                    most_frequent = [rating for rating, count in most_common if count == max_count]

                    #print(f"Calculated rating = {most_frequent}, but the real rating = {test_point.rating}")
                    if most_frequent[0] == test_point.rating:
                        accuracy += 1

                print(f"For k = {k}, accuracy = {accuracy / len(test_set)}")

    print(f"Elapsed time: {(time.time() - start_time):.6f} seconds")


def divide_list(lst, slices_number=5):
    # Calculate the size of each sublist
    sublist_size = len(lst) // slices_number
    remainder = len(lst) % slices_number  # To account for any remaining elements

    lists = []
    start_index = 0

    for i in range(slices_number):
        # If there are remainders, distribute one extra element to the first few sublists
        end_index = start_index + sublist_size + (1 if i < remainder else 0)
        sublist = lst[start_index:end_index]
        lists.append(sublist)
        start_index = end_index

    return lists


if __name__ == "__main__":
    main()
