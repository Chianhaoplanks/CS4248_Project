import string
import re

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

gpu_status = spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])
lemmatizer = nlp.get_pipe("lemmatizer")
punctuation = string.punctuation
stopwords = set(STOP_WORDS)


def restrict_labels(labels, threshold=5):
    defined_labels = []
    for score in labels:
        if score < threshold:
            defined_labels.append(0)
        else:
            defined_labels.append(1)

    return defined_labels


def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(X_train, y_train)


def predict(model, X_test):
    ''' TODO: make your prediction here '''
    labels = model.predict(X_test)
    return labels


def tokenize(text):
    tokens = []
    # remove HTML tags and stuff
    regex = re.compile(r"[A-Za-z0-9\-\'\"?!.,:;()]+") # should we remove punct?
    cleaned_text = ' '.join(re.findall(regex, text.replace("<br", "")))
    for token in nlp(cleaned_text):
        tokens.append(token.lemma_.lower())
    
    return tokens


if __name__ == "__main__":
    print("GPU activated?", gpu_status)
    train = pd.read_csv('aclImdb/train_collated.csv')
    x_train = train['Text']
    y_train = train['Score']
    y_train = restrict_labels(y_train)

    # model = Pipeline([('vec', CountVectorizer(lowercase=False, tokenizer=spacy_tokenizer)), ('tfidf', TfidfTransformer()), ('mnb', BernoulliNB())])
    model = Pipeline([
        ('vec', TfidfVectorizer(lowercase=False, tokenizer=tokenize)),
        ('mnb', BernoulliNB())
    ])

    train_model(model, x_train, y_train)
    y_pred = predict(model, x_train)

    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    score = f1_score(y_train, y_pred)
    print('score on validation for training set: recall=', recall, " precision=", precision, " f1-score=", score)

    test = pd.read_csv('aclImdb/test_collated.csv')
    x_test = test['Text']
    y_test = test['Score']
    y_test = restrict_labels(y_test)
    y_pred = predict(model, x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('score on validation for test set: recall =', recall, " precision =", precision, " f1-score =", score)  # 0.432353 match with test data

