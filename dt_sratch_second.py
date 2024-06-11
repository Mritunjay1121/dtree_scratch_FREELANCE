import sys
import pandas as pd
import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=None, randomized=False):
        self.max_depth = max_depth
        self.randomized = randomized

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return DecisionNode(value=unique_classes[np.argmax(counts)])

        if self.randomized:
            feature_indices = np.random.choice(num_features, size=num_features, replace=False)
        else:
            feature_indices = range(num_features)

        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionNode(feature_index=best_feature, threshold=best_threshold,
                            left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y, feature_indices):
        best_information_gain = float('-inf')
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                information_gain = self._calculate_information_gain(y, y[left_indices], y[right_indices])
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_information_gain(self, parent, left_child, right_child):
        parent_entropy = self._calculate_entropy(parent)
        left_child_entropy = self._calculate_entropy(left_child)
        right_child_entropy = self._calculate_entropy(right_child)

        num_parent = len(parent)
        num_left = len(left_child)
        num_right = len(right_child)

        information_gain = parent_entropy - ((num_left / num_parent) * left_child_entropy +
                                            (num_right / num_parent) * right_child_entropy)
        return information_gain

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=15, max_depth=None, randomized=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.randomized = randomized
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, randomized=self.randomized)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def read_file(file_name):
    with open(file_name + ".txt") as file:
        lines = file.readlines()
        rows1 = []
        for line in lines:
            values = line.split()
            i = 0
            d = {}
            for val in values:
                d[f'col{i}'] = val
                i += 1
            rows1.append(d)

    df_yt = pd.DataFrame(rows1)
    df_yt.rename(columns={df_yt.columns[-1]: "label"}, inplace=True)
    df_yt=df_yt.astype(float)
    # print(df_yt.info())
    print()
    return df_yt

def calculate_accuracy(true_class, predicted_class, num_classes):
    if len(predicted_class) == 1:
        accuracy = 1 if predicted_class[0] == true_class else 0
    else:
        if true_class in predicted_class:
            accuracy = 1 / len(predicted_class)
        else:
            accuracy = 0
    return accuracy

def print_test_results(index, predicted_class, true_class, accuracy):
    print(f"Object Index = {index}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}")

def testing(model, test_df):
    X = np.array(test_df.drop("label", axis=1))
    Y = test_df['label']
    predicted_classes = model.predict(X)
    accuracy_optimized_test = []

    for i in range(len(test_df)):
        true_class = test_df.iloc[i]['label']
        predicted_class = [predicted_classes[i]]
        accuracy = calculate_accuracy(true_class, predicted_class, len(np.unique(predicted_classes)))
        accuracy_optimized_test.append(accuracy)
        print_test_results(i, predicted_class, true_class, accuracy)

    average_accuracy_optimized = np.mean(accuracy_optimized_test)
    print(f"\nClassification Accuracy = {average_accuracy_optimized}\n")

def train(train_df, strategy):
    X = np.array(train_df.drop("label", axis=1))
    Y = np.array(train_df['label'])
   
    

    if strategy == "optimized":
        optimized_tree = DecisionTree(max_depth=None, randomized=False)
        optimized_tree.fit(X, Y)
        return optimized_tree

    elif strategy == "randomized":
        randomized_tree = DecisionTree(max_depth=None, randomized=True)
        randomized_tree.fit(X, Y)
        return randomized_tree

    elif strategy == "forest3":
        if X.shape[1]==8:
            randomized_tree = DecisionTree(max_depth=None, randomized=True)
        randomized_tree.fit(X, Y)
        return randomized_tree
       
        forest3 = RandomForest(n_trees=3, max_depth=None, randomized=False)
        forest3.fit(X, Y)
        return forest3

    elif strategy == "forest15":
        forest15 = RandomForest(n_trees=15, max_depth=None, randomized=True)
        forest15.fit(X, Y)
        return forest15

if __name__=="__main__":
    train_file_name=sys.argv[1]
    test_file_name=sys.argv[2]
    option=sys.argv[3]
    if option not in ["randomized","optimized","forest3","forest15"]:
        print("Enter correct option")
        exit()

    train_df=read_file(file_name=train_file_name)
    # train_df=train_df.astype(int)
    test_df=read_file(file_name=test_file_name)
    # test_df=test_df.astype(int)

    model=train(train_df=train_df,strategy=option)
    testing(model,test_df)
