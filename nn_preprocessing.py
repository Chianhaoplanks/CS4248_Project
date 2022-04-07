import pandas as pd
import numpy as np
import re
import gensim
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding


from rnn import run_nn
from lg_imbd_dataset import convert_sentiment
sampler = RandomUnderSampler(random_state=0)
ohe = OneHotEncoder()

def clean_text(text):
# Convert to lowercase
	text = text.lower()
# Remove punctuation
	text = re.sub(r"[^\w \s]", "", text)
	text = re.sub("\'", "", text)
	return text 
def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def detokenize(text):
	return TreebankWordDetokenizer().detokenize(text)		

def simplify_score(score):
	score = int(score)
	if score > 5:
		return 1
	else:
		return 0


if __name__ == '__main__':
	train = pd.read_csv("./aclImdb/train_collated.csv")
	test = pd.read_csv("IMDB_Dataset.csv")
	y_train = train['Score'].apply(lambda y : simplify_score(y))
	X_train, y_train = sampler.fit_resample(train[['Text']], y_train)
	y = ohe.fit_transform(y_train.values.reshape(-1, 1)).toarray()
	y_train = pd.DataFrame(y)
	print(y_train.head()) 
	temp = []
	X_list = X_train.values.tolist()
	for i in range(len(X_list)):
		temp.append(clean_text(X_list[i][0]))
	data_words = list(sent_to_words(temp))	
	data = []
	
	for i in range(len(data_words)):
		data.append(detokenize(data_words[i]))

	max_words = 5000
	max_len = 200
	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(data)
	sequences = tokenizer.texts_to_sequences(data)
	reviews = pad_sequences(sequences, maxlen=max_len, padding='post')
	print(reviews)
# Repeat for Test set
	y_test = test['sentiment'].apply(lambda y : convert_sentiment(y))
	yt = ohe.fit_transform(y_test.values.reshape(-1, 1)).toarray()
	y_test = pd.DataFrame(yt)
	print(y_test.head())
	
	#X_test, y_test = sampler.fit_resample(test[['Text']], y_test)
	temp = []
	X_test_list = test[['review']].values.tolist()
	for i in range(len(X_test_list)):
		temp.append(clean_text(X_test_list[i][0]))
	data_words = list(sent_to_words(temp))
	test_data = []
	for i in range(len(data_words)):
		test_data.append(detokenize(data_words[i]))
	tokenizer_test = Tokenizer(num_words=max_words)
	tokenizer_test.fit_on_texts(test_data)
	test_seq = tokenizer.texts_to_sequences(test_data)
	test_reviews = pad_sequences(test_seq, maxlen=max_len, padding='post')
	print(test_reviews)
	history = run_nn(reviews, y_train, test_reviews, y_test, max_words, max_len)
	precision = np.mean(history.history["precision"])
	recall = np.mean(history.history["recall"])
	f1 = 2 * (precision * recall) / (precision + recall)
	print(precision, recall, f1)
	print(history.history)
