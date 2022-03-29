import re

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sampler = RandomUnderSampler(random_state=0)
max_features = 6000
embed_size = 128
maxlen = 130
batch_size = 100
epochs = 5
tokenizer = Tokenizer(num_words=max_features)

def restrict_labels(score, threshold=5):
    if score < threshold:
        return 0
    return 1

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stopwords]
    text = " ".join(text)
    return text

def predict(model, X_test):
    ''' TODO: make your prediction here '''
    labels = model.predict(X_test)
    return labels

if __name__ == "__main__":
    train = pd.read_csv('aclImdb/train_collated.csv')
    train['Text'] = train['Text'].apply(clean_text)
    print("Done?")
    train['Score'] = train['Score'].apply(restrict_labels)
    train_bal, train_bal['Score'] = sampler.fit_resample(train[['Text']], train['Score']) 

    x_train = train['Text']
    y_train = train['Score']
    x_train_bal = train_bal['Text']
    y_train_bal = train_bal['Score']
    #x_train_bal = train['Text']
    #y_train_bal = train['Score']
    tokenizer.fit_on_texts(x_train_bal)
    list_tokenized_train = tokenizer.texts_to_sequences(x_train_bal)
    x_train_bal = pad_sequences(list_tokenized_train, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, embed_size))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_bal, y_train_bal, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    test = pd.read_csv('aclImdb/test_collated.csv')
    test['Text'] = test['Text'].apply(clean_text)
    test['Score'] = test['Score'].apply(restrict_labels)
    x_test = test['Text']
    list_tokenized_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    y_test = test['Score']
    y_pred = model.predict(x_test)
    y_pred = (y_pred >= 0.5)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('score on validation for test set: recall =', recall, " precision =", precision, " f1-score =", score)
    print('confusion matrix:', confusion_matrix(y_test, y_pred))

