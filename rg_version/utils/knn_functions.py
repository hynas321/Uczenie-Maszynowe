from itertools import chain


def create_folds(data, num_folds=5):

    fold_size = len(data) // num_folds
    remainder = len(data) % num_folds

    folds = []
    start = 0
    for i in range(num_folds):
        # Calculate the size for this fold (add 1 if remainder > 0)
        current_fold_size = fold_size + (1 if i < remainder else 0)
        folds.append(data[start:start + current_fold_size])
        start += current_fold_size

    # Generate train-test splits for each fold
    for i in range(num_folds):
        # Test fold is the current fold
        test_data = folds[i]

        # Training data is all other folds combined
        train_data = list(chain.from_iterable(folds[:i] + folds[i + 1:]))

        yield train_data, test_data


# Example usage
data = list(range(63))  # Example list with a length not divisible by 5

# Iterate through each fold's train-test split
for fold_num, (train_data, test_data) in enumerate(create_folds(data), 1):
    print(f"Fold {fold_num}:")
    print("Train data:", train_data)
    print("Test data:", test_data)
    print()
