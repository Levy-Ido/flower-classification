import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """
    Load iris dataset from file into a pandas dataframe.

    Returns:
    - data (pd.DataFrame): The iris dataset with feature and target columns.
    """
    data = pd.read_csv('../data/iris.data', header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return data


def split(data):
    """
    Split the dataset into training and testing sets.

    Args:
    - data (pd.DataFrame): The iris dataset with feature and target columns.

    Returns:
    - feat_train (pd.DataFrame): The feature data for training the model.
    - feat_test (pd.DataFrame): The feature data for testing the model.
    - tar_train (pd.Series): The target data for training the model.
    - tar_test (pd.Series): The target data for testing the model.
    """
    feat = data.iloc[:, :-1]
    tar = data.iloc[:, -1]
    feat_train, feat_test, tar_train, tar_test = train_test_split(feat, tar, test_size=0.3, random_state=42)
    return feat_train, feat_test, tar_train, tar_test


def train(feat_train, tar_train):
    """
    Train the logistic regression model.

    Args:
    - feat_train (pd.DataFrame): The feature data for training the model.
    - tar_train (pd.Series): The target data for training the model.

    Returns:
    - model (LogisticRegression): The trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(feat_train, tar_train)
    return model


def print_accuracy():
    """
    Prints the accuracy of a trained logistic regression model.

    Returns:
    None
    """
    data = load_data()
    feat_train, feat_test, tar_train, tar_test = split(data)
    model = train(feat_train, tar_train)
    prediction = model.predict(feat_test)
    accuracy = accuracy_score(tar_test, prediction)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    print_accuracy()
