import os
import re
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import RNN, GRU, LSTM, Dense, Input, Embedding, Dropout, Activation, concatenate
from tensorflow.keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

from embeddings import load

X_train, X_test, y_train, y_test, word_matrix, vocab_size = load()


embed_size = 300
max_len = 5 #200


image_input = Input(shape=(max_len, ))
X = Embedding(vocab_size, embed_size, weights=[word_matrix])(image_input)
X = Bidirectional(GRU(300, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(X)
# Dropout and R-Dropout sequence, inspired by Deep Learning with Python - Francois Chollet
avg_pl = GlobalAveragePooling1D()(X)
max_pl = GlobalMaxPooling1D()(X)
conc = concatenate([avg_pl, max_pl])
X = Dense(7, activation="softmax")(conc)
model = Model(inputs=image_input, outputs=X)


# In[22]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[23]:


saved_model = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5)
callbacks_list = [checkpoint, early]


# In[24]:


batch_sz = 100
epoch = 10
history = model.fit(X_train, y_train, batch_size=batch_sz, epochs=epoch, validation_data=(X_test, y_test), callbacks=callbacks_list)


# In[25]:
model.load_weights(saved_model)

test_values = model.predict([X_test])

y_pred = np.argmax(test_values,axis=1)

from sklearn.metrics import confusion_matrix, classification_report, f1_score

print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average=None))



#print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
import seaborn as sn
import pandas as pd

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

print(val_acc,acc)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title("Model Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title("Model Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

matrix1 = confusion_matrix(y_test, y_pred)

class_names = ["0", "1", "2", "3", "4", "5", "6"]
df_cm1 = pd.DataFrame(matrix1, index = class_names,
                  columns = class_names)

f, (ax1) = plt.subplots(1, 1)

#, (ax4, ax5, ax6),(ax7, ax8, ax9))

sn.heatmap(df_cm1, annot=True, ax=ax1).set_title('Proposed Model')


plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig('CM.pdf')
plt.show()



