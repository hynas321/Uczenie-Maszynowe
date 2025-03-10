from itertools import chain


def create_folds(data, num_folds=5):

    fold_size = len(data) // num_folds
    remainder = len(data) % num_folds

    folds = []
    start = 0
    for i in range(num_folds):
        current_fold_size = fold_size + (1 if i < remainder else 0)
        folds.append(data[start:start + current_fold_size])
        start += current_fold_size

    for i in range(num_folds):
        test_data = folds[i]

        train_data = list(chain.from_iterable(folds[:i] + folds[i + 1:]))

        yield train_data, test_data


data = list(range(63))

for fold_num, (train_data, test_data) in enumerate(create_folds(data), 1):
    print(f"Fold {fold_num}:")
    print("Train data:", train_data)
    print("Test data:", test_data)
    print()
