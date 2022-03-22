import pandas as pd
import re
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import Embedding

nltk.download('stopwords')
nltk.download('punkt')


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Expand contractions
    text = re.sub(r"'s", r" is", text)
    text = re.sub(r"'ve", r" have", text)
    text = re.sub(r"'re'", r"  are", text)
    text = re.sub(r"can't'", r" cannot", text)
    text = re.sub(r"n't", r" not", text)
    text = re.sub(r"'m", r" am", text)
    # Remove punctuation
    text = re.sub(r"[^\w \s]", "", text)
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    data = [word for word in tokens if word not in nltk.corpus.stopwords.words("english")]
    return " ".join(data)


def simplify_score(score):
    score = int(score)
    if score >= 1 and score <= 5:
        return -1
    elif score >= 6 and score <= 10:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train = pd.read_csv("./aclImdb/train_collated.csv")
    X_train = train['Text'].apply(lambda x: clean_text(x))
    y_train = train['Score'].apply(lambda y : simplify_score(y))
    max_words = 5000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    reviews = pad_sequences(sequences, maxlen=max_len)
    embedding_layer = Embedding(1000, 64)