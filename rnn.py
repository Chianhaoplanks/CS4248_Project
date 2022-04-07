from keras.metrics import Precision, Recall
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

def get_metrics(y_true, y_pred):
	true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	#Recall
	pos = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_pos / (pos + K.epsilon())
	#Precision
	pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_pos / (pred_pos + K.epsilon())
	#F1
	f1 = 2*((precision * recall)/(precision + recall + K.epsilon()))
	return precision, recall, f1

def run_nn(X_train, y_train, X_test, y_test, max_words, max_len):
	model = Sequential()
	model.add(layers.Embedding(max_words, 20, input_length=max_len))
	#model.add(layers.Flatten())
	model.add(layers.Bidirectional(layers.LSTM(15, dropout=0.5)))
	model.add(layers.Dense(2, activation='sigmoid'))
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[Precision(), Recall()])
	print(model.summary())
	checkpoint = ModelCheckpoint("best_model.hdf5", verbose=1, save_best_only=True, mode='auto', save_freq=1, save_weights_only=False)
	history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
	#y_pred = model.predict(X_test)
	#precision, recall, f1 = get_metrics(y_test, y_pred)
	return history
