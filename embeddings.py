import os
import io
import csv
import re,string
import numpy as np
import gensim
from gensim.models import FastText
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
import fasttext

data_dir = "/home/reddy/SAKSHI/Dataset/"

def load():
	headlines = []
	stories = []
	labels = [] #
	
	for category in os.listdir(data_dir):
		print(category)
		for item in os.listdir(data_dir + category):
			f = open(data_dir + category + "/" + str(item), "r")
			texts = f.readlines()
			temp = ""
			headlines.append(str(texts[0]))
			for i in range(1, len(texts)):
				temp = temp + " " + str(texts[i].strip())		
			stories.append(temp)
	
			if category == "economy":
				labels.append(0)
			elif category == "corporate":
				labels.append(1)
			elif category == "market":
				labels.append(2)  
			elif category == "movies":
				labels.append(3)
			elif category == "politics":
				labels.append(4)
			elif category == "technology":
				labels.append(5)
			elif category == "sports":
				labels.append(6)
	
	
	target_names = ["economy", "corporate", "market", "movies", "politics", "technology", "sports"]
	
	class_names = ["0", "1", "2", "3", "4", "5", "6"]
	
	tokenizer = Tokenizer(oov_token="<OOV>") #oov_token="<OOV>"
	tokenizer.fit_on_texts(headlines)
	vocabulary = tokenizer.word_index
	
	
	#print(vocabulary)
	
	X_train, X_test, y_train, y_test = train_test_split(headlines, labels, test_size=0.2, shuffle=True, random_state=42)
	
	train_sequences = tokenizer.texts_to_sequences(X_train)
	val_sequences = tokenizer.texts_to_sequences(X_test)
	
	train_tweet = pad_sequences(train_sequences,padding="post",truncating="post",maxlen=5)
	test_tweet = pad_sequences(val_sequences,padding="post",truncating="post",maxlen=5)
		
	
	
	vocab_size = len(vocabulary)+1	
	b = 0
	c = 0
	size = 300
	
	word_matrix = np.zeros((len(vocabulary)+1, size)) 
		
	print("Creating word matrix....")
		
	print("vocab_size: ", vocab_size)
	
	#fin = io.open('cc.te.300.bin', 'r', encoding='utf-8', newline='\n', errors='ignore')
	model = fasttext.load_model('indicnlp.ft.te.300.bin')	
	#n, d = map(int, fin.readline().split())
	#data = {}
	#for line in model:
	#    tokens = line.rstrip().split(' ')
	#    data[tokens[0]] = map(float, tokens[1:])
	
		
	print("Creating word matrix....")
		
	print("vocab_size: ", vocab_size)
	
	for word, i in vocabulary.items():
		try:
			word_matrix[i] = model.get_word_vector(word) #data[word] 
			c+=1
		except KeyError:
			# if a word is not include in the vocabulary, it's word embedding will be set by random.
			word_matrix[i] = np.random.uniform(-0.25,0.25,size)
			b+=1	
	

	print('there are %d words in model'%c)		
	print('there are %d words not in model'%b)
	np.ndarray.dump(word_matrix, open('word_matrix.np', 'wb'))

	train_tweet = np.array(train_tweet)
	test_tweet = np.array(test_tweet)
	y_train = np.array(y_train)	
	y_test = np.array(y_test)

	return train_tweet, test_tweet, y_train, y_test, word_matrix, vocab_size
