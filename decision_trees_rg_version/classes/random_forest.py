import numpy as np
from typing import List, Optional
from decision_trees_rg_version.classes.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, min_samples_split: int = 2, max_features: Optional[int] = None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees: List[DecisionTree] = []  # List to store individual trees

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_samples, n_features = X.shape

        # If max_features is not provided, use sqrt of total features
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Create and train a new tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:

        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])  # Shape: (n_trees, n_samples)

        # Aggregate predictions using majority voting
        def majority_vote(predictions):
            counts = np.bincount(predictions)
            return np.argmax(counts)

        # Apply majority voting for each sample
        return np.apply_along_axis(majority_vote, axis=0, arr=tree_predictions)

