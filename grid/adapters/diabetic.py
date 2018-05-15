from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
import sys
import time
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm

img_path = f"/home/hung/.kaggle/competitions/diabetic-retinopathy-detection/train"
train_file = f"/home/hung/.kaggle/competitions/diabetic-retinopathy-detection/trainLabels.csv"
num_classes = 5
img_size = 150

def load_data():
    df_train = pd.read_csv(train_file)
    df_train.head()
    df_train['image'] = df_train['image'].astype(str)
    df_train['level'] = df_train['level'].astype(int)

    x = []
    y = []

    # Processing image files
    #num_images = len(df_train['image'])
    num_images = 2000
    for i in tqdm(range(num_images)):
        img_file = img_path + '/' + df_train['image'][i] + '.jpeg'
        img_data = load_img(img_file, target_size=(img_size, img_size))
        img_data = img_to_array(img_data)
        img_data.reshape((1,) + img_data.shape)

        x.append(img_data)
        y.append(df_train['level'][i])

        time.sleep(0.1)

    x = np.asarray(x)
    y = np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=0.2,
                                                        stratify = y)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test_samples')

    return (x_train, y_train), (x_test, y_test)