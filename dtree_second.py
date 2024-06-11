
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def read_data_from_file(file_name):
    """
    Read data from a text file and convert it into a DataFrame.

    Parameters:
    - file_name: Name of the text file (without extension)

    Returns:
    - df: DataFrame containing the data
    """
    with open(file_name + ".txt") as file:
        lines = file.readlines()
        rows = []

        # Process each line
        for line in lines:
            # Split values based on spaces
            values = line.split()

            # Convert values into a dictionary
            row_dict = {f'col{i}': val for i, val in enumerate(values)}
            rows.append(row_dict)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(rows)

        # Rename the last column to "label"
        df.rename(columns={df.columns[-1]: "label"}, inplace=True)

    return df
        


def training(df,option):
    X_train = df.drop('label', axis=1)
    Y_train = df['label']

    if option=="optimized":
        optimized_obj = DecisionTreeClassifier()
        optimized_obj.fit(X_train, Y_train)
        return optimized_obj

    elif option=="randomized":
        randomized_obj = DecisionTreeClassifier(splitter='random')
        randomized_obj.fit(X_train, Y_train)
        return randomized_obj

    elif option=="forest3":
        rf3_obj = RandomForestClassifier(n_estimators=3, random_state=42)
        rf3_obj.fit(X_train, Y_train)
        return rf3_obj

    elif option=="forest15":
        rf15_obj = RandomForestClassifier(n_estimators=15, random_state=42)
        rf15_obj.fit(X_train, Y_train)

        return rf15_obj


def calculate_accuracy(true_class, predicted_class):
    """
    Calculate accuracy for a given true class and predicted class.

    Parameters:
    - true_class: Actual class label
    - predicted_class: List of predicted class labels


    Returns:
    - accuracy: Accuracy value (float)
    """
    if len(predicted_class) == 1:
        # No ties
        accuracy = 1.0 if predicted_class[0] == true_class else 0.0
    else:
        # Ties
        accuracy = 1.0 / len(predicted_class) if true_class in predicted_class else 0.0

    return accuracy


def print_test_results(index, predicted_class, true_class, accuracy):
    """
    Print information for each test object.

    Parameters:
    - index: Index of the test object
    - predicted_class: Predicted class label
    - true_class: True class label
    - accuracy: Accuracy value
    """
    print(f"Object Index = {index}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}")


def testing(model, test_df):
    """
    Perform testing on a given model and print test results.

    Parameters:
    - model: Trained model
    - test_df: DataFrame containing test data and labels
    """
    X = test_df.drop("label", axis=1)
    Y = test_df['label']
    predicted_classes = model.predict(X)
    accuracy_optimized_test = []

    for i, (true_class, predicted_class) in enumerate(zip(Y, predicted_classes)):
        accuracy = calculate_accuracy(true_class, [predicted_class])
        accuracy_optimized_test.append(accuracy)
        print_test_results(i, [predicted_class], true_class, accuracy)

    # Print overall classification accuracy
    average_accuracy_optimized = np.mean(accuracy_optimized_test)
    print(f"\nClassification Accuracy (Optimized) = {average_accuracy_optimized}\n")







if __name__=="__main__":
    train_file_name=sys.argv[1]
    test_file_name=sys.argv[2]
    option=sys.argv[3]
    

    train_df=read_data_from_file(file_name=train_file_name)
    test_df=read_data_from_file(file_name=test_file_name)
    model=training(df=train_df,option=option)
    testing(model,test_df)

    # print(read_file(file_name=train_file_name))
    # print("----------------------------------------------------------")
    # print(read_testfile(file_name=test_file_name))

    
