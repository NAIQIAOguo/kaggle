import pandas as pd
import re
import numpy as np

def load(file_name):
    df_test = pd.read_csv(file_name)
    sLength = len(df_test['id'])
    df_test.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_test = df_test.fillna("miss")
    df_test['name'] = df_test['name'].str.lower()
    df_test['lvl1'] = df_test['lvl1'].str.lower()
    df_test['lvl2'] = df_test['lvl2'].str.lower()
    df_test['lvl3'] = df_test['lvl3'].str.lower()
    df_test['descrption'] = df_test['descrption'].str.lower()
    df_test['type'] = df_test['type'].str.lower()
    df_test['score'] = pd.Series(np.zeros(sLength, dtype=np.int), index=df_test.index)
    return df_test

def loadY(file_name):
    df = pd.read_csv(file_name)
    return df

def loadX(file_name):
    df_train = pd.read_csv(file_name)
    df_label = pd.read_csv('train_label.csv')
    df_train = df_train.set_index('id').join(df_label.set_index('id'), how='inner')
    df_train.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_train['name'] = df_train['name'].str.lower()
    df_train['lvl1'] = df_train['lvl1'].str.lower()
    df_train['lvl2'] = df_train['lvl2'].str.lower()
    df_train['lvl3'] = df_train['lvl3'].str.lower()
    df_train['descrption'] = df_train['descrption'].str.lower()
    df_train['type'] = df_train['type'].str.lower()
    df_train = df_train.fillna("miss")
    return df_train

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

X_train =loadX('train_data.csv')
X_test = load('test_data.csv')
df_train = pd.DataFrame(data=X_train['lvl1']+X_train['lvl2']+X_train['lvl3']+X_train['type']+X_train['name']+X_train['descrption'],columns=['train'])

X_train_text = df_train.train
print(X_train_text.values)

df_test = pd.DataFrame(data=X_test['name']+X_test['lvl1']+X_test['lvl2']+X_test['lvl3']+X_test['type']+X_test['descrption'],columns=['test'])
X_test_text = df_test.test
# print(X_test_text.values)
print(X_test_text.shape)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train_text)

X_train_hh = tokenizer.texts_to_sequences(X_train_text)
X_train_hh = pad_sequences(X_train_hh,maxlen=256)
print(X_train_hh)
print(X_train_hh.shape)

X_test_hh = tokenizer.texts_to_sequences(X_test_text)
X_test_hh = pad_sequences(X_test_hh,maxlen=256)
print(X_test_hh)
print(X_test_hh.shape)

y_score = X_train.score.values
print(y_score)
print(y_score.shape)

import keras
import tensorflow as tf

model = keras.Sequential()
model.add(keras.layers.Embedding(20000, 16))

model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train_hh[2001:],y_score[2001:],
          epochs=11,
          batch_size=128,
          validation_data=(X_train_hh[0:2000], X_train[0:2000].score.values),
          verbose=2)

output_array = model.predict(X_test_hh)
print(output_array)
df_end_id = pd.DataFrame(data=X_test.id)
print(X_test.id)


df_end_score = pd.DataFrame(data=output_array,columns=['score'])
df_frames = [df_end_id,df_end_score]
df_end = pd.concat(df_frames,axis=1)

df_end.to_csv('output.csv',index=False)
print(df_end)
print(df_end.shape)
