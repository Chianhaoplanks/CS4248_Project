import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS

nlp = English()
stopwords = list(STOP_WORDS)


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

def tokenize(text):
    tokens = []
    for token in nlp(text):
        if token.pos_ == "PROPN":
            tokens.append(token.text)
        else:
            tokens.append(token.lower_)
    return tokens

if __name__ == "__main__":
    train = pd.read_csv('aclImdb/train_collated.csv')
    x_train = train['Text']
    y_train = train['Score']

    y_train = restrict_labels(y_train)
    # model = Pipeline([('vec', TfidfVectorizer()), ('mnb', BernoulliNB())])
    model = Pipeline([('vec', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()), ('mnb', LogisticRegression(max_iter=5000))])

    train_model(model, x_train, y_train)
    y_pred = predict(model, x_train)

    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    score = f1_score(y_train, y_pred)
    print('score on validation for training set: recall=', recall, " precision=", precision, " f1-score=", score)

    test = pd.read_csv('aclImdb/test_collated.csv')
    x_test = test['Text']
    y_test = test['Score']
    y_test = define_labels_with_threshold(y_test)

    y_pred = predict(model, x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('score on validation for test set: recall =', recall, " precision =", precision, " f1-score =", score)  # 0.432353 match with test data


