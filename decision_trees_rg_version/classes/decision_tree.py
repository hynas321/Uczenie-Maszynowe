from collections import Counter
import numpy as np
from typing import Optional, Tuple
from decision_trees_rg_version.classes.tree_node import TreeNode

class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_features: Optional[int] = None):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # Subsampling features
        self.root = None
        self.n_features = None  # Tracks total number of features in the dataset

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.n_features = X.shape[1]

        # Validate max_features
        if self.max_features and self.max_features > self.n_features:
            raise ValueError(
                f"`max_features` ({self.max_features}) cannot be greater than the total number of features ({self.n_features})."
            )

        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:

        num_samples, num_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        # Select a subset of features if max_features is specified
        if self.max_features:
            feature_indices = np.random.choice(self.n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(self.n_features)

        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        # Split data into left and right subsets
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the node
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_feature, best_threshold = None, None
        best_gain = -float('inf')
        current_loss = self._calculate_loss(y)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_loss = self._calculate_loss(y[left_indices])
                right_loss = self._calculate_loss(y[right_indices])
                weighted_loss = (len(y[left_indices]) / len(y)) * left_loss + \
                                (len(y[right_indices]) / len(y)) * right_loss

                gain = current_loss - weighted_loss
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_loss(self, y: np.ndarray) -> float:

        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        counter = Counter(y)
        return counter.most_common(1)[0][0]  # Mode (most frequent rating)

    def _traverse_tree(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf_node():
            return int(round(node.value))
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def visualize(self, node: Optional[TreeNode] = None, depth: int = 0):
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{'  ' * depth}Leaf: {node.value}")
        else:
            print(f"{'  ' * depth}[X{node.feature} <= {node.threshold}]")
            self.visualize(node.left, depth + 1)
            self.visualize(node.right, depth + 1)
