import string
import re

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
punctuation = string.punctuation
stopwords = set(stopwords.words('english'))
undersampler = RandomUnderSampler(random_state=0)


def convert_sentiment(sentiment):
    if sentiment == 'positive':
        return 1
    else:
        return 0


def restrict_labels(score, threshold=5):
    if score < threshold:
        return 0
    return 1


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
    regex = re.compile(r"[A-Za-z0-9\-]+")
    cleaned_text = re.findall(regex, text.replace("<br", ""))
    for token in cleaned_text:
        tokens.append(lemmatizer.lemmatize(token).lower())

    return tokens


if __name__ == "__main__":
    train = pd.read_csv('aclImdb/train_collated.csv')
    train['Score'] = train['Score'].apply(restrict_labels)

    x_train = train['Text']
    y_train = train['Score']

    model = Pipeline([
        ('vec', TfidfVectorizer(lowercase=False, tokenizer=tokenize, ngram_range=(1, 2))),
        ('mnb', LogisticRegression(max_iter=5000))
    ])

    train_model(model, x_train, y_train)
    y_pred = predict(model, x_train)

    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    score = f1_score(y_train, y_pred)
    print('score on validation for training set: recall =', recall, "precision =", precision, "f1-score =", score)

    test = pd.read_csv('IMDB_Dataset.csv')
    x_test = test['review']
    y_test = test['sentiment'].apply(convert_sentiment)
    y_pred = predict(model, x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('score on validation for test set: recall =', recall, "precision =", precision, "f1-score =",
          score)  # 0.432353 match with test data

