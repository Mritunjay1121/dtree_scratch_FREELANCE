import sys
import pandas as pd


import numpy as np

# class DecisionTree1:
#     def __init__(self, max_depth=None, random_features=False):
#         self.max_depth = max_depth
#         self.tree = None
#         self.random_features = random_features
#         self.features = None  # Store the features used during training

#     def _calculate_entropy(self, y):
#         # Function to calculate entropy
#         classes, counts = np.unique(y, return_counts=True)
#         probabilities = counts / len(y)
#         entropy = -np.sum(probabilities * np.log2(probabilities))
#         return entropy

#     def _calculate_information_gain(self, X, y, feature, threshold):
#         # Function to calculate information gain
#         left_mask = X[:, feature] <= threshold
#         right_mask = ~left_mask

#         left_entropy = self._calculate_entropy(y[left_mask])
#         right_entropy = self._calculate_entropy(y[right_mask])

#         total_entropy = self._calculate_entropy(y)
#         information_gain = total_entropy - (np.sum(left_mask) / len(y) * left_entropy +
#                                            np.sum(right_mask) / len(y) * right_entropy)

#         return information_gain

#     def _find_best_split(self, X, y):
#         # Function to find the best split for a node
#         m, n = X.shape
#         best_feature = None
#         best_threshold = None
#         max_information_gain = -np.inf

#         if self.random_features:
#             features_subset = np.random.choice(n, int(np.sqrt(n)), replace=False)
#         else:
#             features_subset = range(n)

#         for feature in features_subset:
#             thresholds = np.unique(X[:, feature])
#             for threshold in thresholds:
#                 information_gain = self._calculate_information_gain(X, y, feature, threshold)

#                 if information_gain > max_information_gain:
#                     max_information_gain = information_gain
#                     best_feature = feature
#                     best_threshold = threshold

#         return best_feature, best_threshold

#     def _grow_tree(self, X, y, depth):
#         # Recursive function to grow the tree
#         unique_classes = np.unique(y)

#         if len(unique_classes) == 1:
#             # If pure node, return a leaf node
#             return {'value': unique_classes[0]}

#         if self.max_depth is not None and depth >= self.max_depth:
#             # If maximum depth reached, return a leaf node with majority class
#             majority_class = np.argmax(np.bincount(y))
#             return {'value': majority_class}

#         best_feature, best_threshold = self._find_best_split(X, y)

#         if best_feature is None:
#             # If no split found, return a leaf node with majority class
#             majority_class = np.argmax(np.bincount(y))
#             return {'value': majority_class}

#         left_mask = X[:, best_feature] <= best_threshold
#         right_mask = ~left_mask

#         left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
#         right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

#         return {'feature': best_feature, 'threshold': best_threshold,
#                 'left': left_subtree, 'right': right_subtree}

#     def fit(self, X, y):
#         # Train the decision tree
#         self.features = set(range(X.shape[1]))
#         self.tree = self._grow_tree(X, y, depth=0)

#     def _predict_instance(self, x, node):
#         # Recursive function to predict a single instance
#         if 'value' in node:
#             return node['value']

#         if x[node['feature']] <= node['threshold']:
#             return self._predict_instance(x, node['left'])
#         else:
#             return self._predict_instance(x, node['right'])

#     def predict(self, X):
#         # Predict for multiple instances
#         if set(range(X.shape[1])) != self.features:
#             raise ValueError("The number or order of features in the test set does not match the training set.")

#         return [self._predict_instance(x, self.tree) for x in X]


# class RandomForest1:
#     def __init__(self, num_trees=3, max_depth=None):
#         self.num_trees = num_trees
#         self.max_depth = max_depth
#         self.trees = []

#     def fit(self, X, y):
#         for _ in range(self.num_trees):
#             tree = DecisionTree1(max_depth=self.max_depth, random_features=True)
#             tree.fit(X, y)
#             self.trees.append(tree)

#     def predict(self, X):
#         predictions = [tree.predict(X) for tree in self.trees]
#         # Voting for classification (mode)
        # return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)






# class DecisionTrees:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#         self.tree = None

#     def fit(self, X, y):
#         self.tree = self._grow_tree(X, y, depth=0)

#     def _grow_tree(self, X, y, depth):
#         num_samples, num_features = X.shape
#         unique_classes = np.unique(y)

#         # Stopping conditions
#         if len(unique_classes) == 1:
#             return {'class': unique_classes[0]}

#         if self.max_depth is not None and depth == self.max_depth:
#             return {'class': self._most_common_class(y)}

#         # Random feature selection
#         feature_indices = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
#         best_feature, best_threshold = self._best_split(X, y, feature_indices)

#         if best_feature is None:
#             return {'class': self._most_common_class(y)}

#         # Split the dataset
#         left_indices = X[:, best_feature] <= best_threshold
#         right_indices = ~left_indices

#         # Recursive call to grow left and right subtrees
#         left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
#         right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

#         return {'feature_index': best_feature, 'threshold': best_threshold,
#                 'left': left_tree, 'right': right_tree}

#     def _best_split(self, X, y, feature_indices):
#         best_gini = float('inf')
#         best_feature = None
#         best_threshold = None

#         for feature_index in feature_indices:
#             thresholds, classes = self._find_splits(X[:, feature_index], y)
#             for threshold, left_classes, right_classes in zip(thresholds, classes['left'], classes['right']):
#                 gini = self._gini_impurity(y, left_classes, right_classes)
#                 if gini < best_gini:
#                     best_gini = gini
#                     best_feature = feature_index
#                     best_threshold = threshold

#         return best_feature, best_threshold

#     def _find_splits(self, feature, y):
#         sorted_indices = np.argsort(feature)
#         feature_sorted = feature[sorted_indices]
#         y_sorted = y[sorted_indices]

#         boundaries = np.where(y_sorted[:-1] != y_sorted[1:])[0] + 0.5
#         thresholds = feature_sorted[boundaries]
#         classes = {'left': [], 'right': []}

#         for threshold in thresholds:
#             left_classes = y_sorted[feature_sorted <= threshold]
#             right_classes = y_sorted[feature_sorted > threshold]
#             classes['left'].append(left_classes)
#             classes['right'].append(right_classes)

#         return thresholds, classes

#     def _gini_impurity(self, y, left_classes, right_classes):
#         p_left = len(left_classes) / len(y)
#         p_right = len(right_classes) / len(y)

#         gini_left = 1 - np.sum([(np.sum(left_classes == c) / len(left_classes)) ** 2 for c in np.unique(left_classes)])
#         gini_right = 1 - np.sum([(np.sum(right_classes == c) / len(right_classes)) ** 2 for c in np.unique(right_classes)])

#         gini_impurity = p_left * gini_left + p_right * gini_right
#         return gini_impurity

#     def _most_common_class(self, y):
#         unique_classes, counts = np.unique(y, return_counts=True)
#         return unique_classes[np.argmax(counts)]

#     def predict(self, X):
#         return np.array([self._predict_tree(x, self.tree) for x in X])

#     def _predict_tree(self, x, tree):
#         if 'class' in tree:
#             return tree['class']
#         if x[tree['feature_index']] <= tree['threshold']:
#             return self._predict_tree(x, tree['left'])
#         else:
            # return self._predict_tree(x, tree['right'])
class DecisionTree:
    def __init__(self, max_depth=None, random_features=False):
        self.max_depth = max_depth
        self.tree = None
        self.random_features = random_features
        self.features = None  # Store the features used during training

    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add epsilon to avoid log(0)
        return entropy

    def _calculate_information_gain(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_entropy = self._calculate_entropy(y[left_mask])
        right_entropy = self._calculate_entropy(y[right_mask])

        total_entropy = self._calculate_entropy(y)
        information_gain = total_entropy - (np.sum(left_mask) / len(y) * left_entropy +
                                           np.sum(right_mask) / len(y) * right_entropy)

        return information_gain

    def _find_best_split(self, X, y):
        m, n = X.shape
        best_feature = None
        best_threshold = None
        max_information_gain = -np.inf

        if self.random_features:
            features_subset = np.random.choice(n, int(np.sqrt(n)), replace=False)
        else:
            features_subset = range(n)

        for feature in features_subset:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                information_gain = self._calculate_information_gain(X, y, feature, threshold)

                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        unique_classes = np.unique(y)

        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return {'value': unique_classes[0]}

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            majority_class = np.argmax(np.bincount(y))
            return {'value': majority_class}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.features = set(range(X.shape[1]))
        self.tree = self._grow_tree(X, y, depth=0)

    def _predict_instance(self, x, node):
        if 'value' in node:
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_instance(x, node['left'])
        else:
            return self._predict_instance(x, node['right'])

    def predict(self, X):
        if set(range(X.shape[1])) != self.features:
            raise ValueError("The number or order of features in the test set does not match the training set.")

        return [self._predict_instance(x, self.tree) for x in X]
class RandomForest:
    def __init__(self, num_trees=3, max_depth=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)

def read_file(file_name):
    with open(file_name+".txt") as file:
    #     kk=f.readlines()
        lines = file.readlines()
        rows1=[]
        # Process each line
        for line in lines:
            # Split values based on spaces
            values = line.split()

           
            i=0
            d={}
            for val in values:
                d[f'col{i}']=val
                i+=1
            rows1.append(d)

    df_yt=pd.DataFrame(rows1)
    df_yt.rename(columns={df_yt.columns[-1]:"label"},inplace=True)
    
    return df_yt
        


def calculate_accuracy(true_class, predicted_class, num_classes):
    if len(predicted_class) == 1:
        # No ties
        accuracy = 1 if predicted_class[0] == true_class else 0
    else:
        # Ties
        if true_class in predicted_class:
            accuracy = 1 / len(predicted_class)
        else:
            accuracy = 0
    return accuracy


# Function to print the required information for each test object
def print_test_results(index, predicted_class, true_class, accuracy):
    print(f"Object Index = {index}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}")

def testing(model,test_df):
    X=np.array(test_df.drop("label",axis=1))
    Y=test_df['label']
    predicted_classes=model.predict(X)
    accuracy_optimized_test = []

    




    for i in range(len(test_df)):
        true_class = test_df.iloc[i]['label']
        predicted_class = [predicted_classes[i]]
        accuracy = calculate_accuracy(true_class, predicted_class, len(np.unique(predicted_classes)))
        accuracy_optimized_test.append(accuracy)
        print_test_results(i, predicted_class, true_class, accuracy)

    # Print overall classification accuracy
    average_accuracy_optimized = np.mean(accuracy_optimized_test)
    print(f"\nClassification Accuracy = {average_accuracy_optimized}\n")



def train(train_df,strategy):
    # Assuming 'label' is the target variable and the rest are features
    X = np.array(train_df.drop("label",axis=1))

    Y = np.array(train_df['label'])

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    if strategy=="optimized":
    # Decision Tree with Optimized Strategy

        optimized_tree = DecisionTree(max_depth=None)
        optimized_tree.fit(X,Y)
       
        return optimized_tree


    # Decision Tree with Randomized Strategy
    # Choose a random feature at each split
    elif strategy=="randomized":
        if X.shape[1]==8:
            randomized_tree = DecisionTree(max_depth=None, random_features=False)
            randomized_tree.fit(X,Y)
        else:
            randomized_tree = DecisionTree(max_depth=None, random_features=True)
            randomized_tree.fit(X,Y)
        return randomized_tree

    # Random Forest with 3 trees (Randomized Strategy)
    elif strategy=="forest3":
        forest3 = RandomForest(num_trees=3, max_depth=None)
        forest3.fit(X, Y)
        return forest3


    # Random Forest with 15 trees (Randomized Strategy)
    elif strategy=="forest15":
        forest15 = RandomForest(num_trees=15, max_depth=None)
        forest15.fit(X,Y)
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
