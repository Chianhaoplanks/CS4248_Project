import pandas as pd

from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors._nearest_centroid import NearestCentroid

from sklearn.linear_model import LogisticRegression

def restrict_labels(labels):
    restricted_labels = []
    for score in labels:
        if score > 0:
            restricted_labels.append(1)
        else:
            restricted_labels.append(0)

    return restricted_labels


def define_labels_with_threshold(labels, threshold=6.5):
    defined_labels = []

    for score in labels:
        if score < threshold:
            defined_labels.append(0)
        else:
            defined_labels.append(1)

    return defined_labels


def process_data(data):
    pass


def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(X_train, y_train)


def predict(model, X_test):
    ''' TODO: make your prediction here '''
    labels = model.predict(X_test)
    return labels


if __name__ == "__main__":
    train = pd.read_csv('aclImdb/train_collated.csv')
    x_train = train['Text']
    y_train = train['Score']

    y_train = restrict_labels(y_train)
    model = Pipeline([('vec', TfidfVectorizer()), ('clf', NearestCentroid())])

    train_model(model, x_train, y_train)
    y_pred = predict(model, x_train)

    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation for training set = {}'.format(score))

    test = pd.read_csv('aclImdb/test_collated.csv')
    x_test = test['Text']
    y_test = test['Score']

    y_test = define_labels_with_threshold(y_test)
    y_pred = predict(model, x_test)

    score = f1_score(y_test, y_pred, average='macro')
    print('score on validation for test set = {}'.format(score))  # 0.432353 match with test data


def spacy_tokenizer(text):
    tokens = []
    for token in nlp(text):
        if token.is_stop or token.is_punct:
            continue

        if token.pos_ == "PROPN":
            tokens.append(token.text)
        else:
            tokens.append(token.lower_)

    return tokens