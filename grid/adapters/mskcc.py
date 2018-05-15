from __future__ import print_function
import numpy as np
import pandas as pd
import keras as keras
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

training_text = '/home/hung/.openmined/data/MSKCC/training_text'
training_var = '/home/hung/.openmined/data/MSKCC/training_variants'

def load_data():
    print('Loading data...')
    df_train_txt = pd.read_csv(training_text, sep='\|\|', 
                               header=None, skiprows=1,
                               names=["ID", "Text"])
    df_train_txt.head()
    df_train_txt['ID'] = df_train_txt['ID'].astype(int)
    df_train_txt['Text'] = df_train_txt['Text'].astype(str)

    df_train_var = pd.read_csv(training_var)
    df_train_var.head()
    df_train_var['ID'] = df_train_var['ID'].astype(int)
    df_train_var['Gene'] = df_train_var['Gene'].astype(str)
    df_train_var['Variation'] = df_train_var['Variation'].astype(str)
    df_train_var['Class'] = df_train_var['Class'].astype(int)

    df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
    df_train.head()

    num_words = 500
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df_train['Text'].values)

    x = tokenizer.texts_to_sequences(df_train['Text'].values)
    x = pad_sequences(x, maxlen=num_words)
    y = pd.get_dummies(df_train['Class'].values)

    print('x shape %s' % (x.shape,))
    print('y shape %s' % (y.shape,))
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        stratify = y)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test_samples')

    return (x_train, y_train), (x_test, y_test)