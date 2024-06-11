
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score







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
        

def read_testfile(file_name):
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

def train(train_df,strategy):
    # Assuming 'label' is the target variable and the rest are features
    X = train_df.drop('label', axis=1)
    Y = train_df['label']

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    if strategy=="optimized":
    # Decision Tree with Optimized Strategy
        dt_optimized = DecisionTreeClassifier()
        dt_optimized.fit(X, Y)
        return dt_optimized
    # y_pred_optimized = dt_optimized.predict(X_test)
    # accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    # print(f"Accuracy (Optimized): {accuracy_optimized}")

    # Decision Tree with Randomized Strategy
    # Choose a random feature at each split
    elif strategy=="randomized":
        dt_randomized = DecisionTreeClassifier(splitter='random')
        dt_randomized.fit(X, Y)
        return dt_randomized
    # y_pred_randomized = dt_randomized.predict(X_test)
    # accuracy_randomized = accuracy_score(y_test, y_pred_randomized)
    # print(f"Accuracy (Randomized): {accuracy_randomized}")

    # Random Forest with 3 trees (Randomized Strategy)
    elif strategy=="forest3":
        rf3 = RandomForestClassifier(n_estimators=3, random_state=42)
        rf3.fit(X, Y)
        return rf3
    # y_pred_rf3 = rf3.predict(X_test)
    # accuracy_rf3 = accuracy_score(y_test, y_pred_rf3)
    # print(f"Accuracy (Forest3): {accuracy_rf3}")

    # Random Forest with 15 trees (Randomized Strategy)
    elif strategy=="forest15":
        rf15 = RandomForestClassifier(n_estimators=15, random_state=42)
        rf15.fit(X, Y)
        return rf15
    # y_pred_rf15 = rf15.predict(X_test)
    # accuracy_rf15 = accuracy_score(y_test, y_pred_rf15)
    # print(f"Accuracy (Forest15): {accuracy_rf15}")

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
    X=test_df.drop("label",axis=1)
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
    print(f"\nClassification Accuracy (Optimized) = {average_accuracy_optimized}\n")











if __name__=="__main__":
    train_file_name=sys.argv[1]
    test_file_name=sys.argv[2]
    option=sys.argv[3]
    

    train_df=read_file(file_name=train_file_name)
    test_df=read_testfile(file_name=test_file_name)
    model=train(train_df=train_df,strategy=option)
    testing(model,test_df)

    # print(read_file(file_name=train_file_name))
    # print("----------------------------------------------------------")
    # print(read_testfile(file_name=test_file_name))

    
