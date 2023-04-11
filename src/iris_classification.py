import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    data = pd.read_csv('../data/iris.data', header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return data


def split(data):
    feat = data.iloc[:, :-1]
    tar = data.iloc[:, -1]
    feat_train, feat_test, tar_train, tar_test = train_test_split(feat, tar, test_size=0.3, random_state=42)
    return feat_train, feat_test, tar_train, tar_test


def train(feat_train, tar_train):
    model = LogisticRegression()
    model.fit(feat_train, tar_train)
    return model


if __name__ == "__main__":
    data = load_data()
    feat_train, feat_test, tar_train, tar_test = split(data)
    model = train(feat_train, tar_train)
    prediction = model.predict(feat_test)
    accuracy = accuracy_score(tar_test, prediction)
    print(f"Accuracy: {accuracy}")