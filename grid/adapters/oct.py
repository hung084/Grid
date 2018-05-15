
from __future__ import print_function
import numpy as np
import keras as keras
import itertools
import skimage
import scipy
import cv2
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.applications.vgg16 import VGG16

from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split


# Set parameters:
batch_size = 32
num_classes = 4
epochs = 83
imageSize = 150
test_dir = "/home/hung/.openmined/data/OCT2017/test/"

def extract_data(directory):
    x = []
    y = []
    label = 4
    
    for dirName in os.listdir(directory):
        if not dirName.startswith('.'):
            if dirName in ['NORMAL']:
                label = 0
            elif dirName in ['CNV']:
                label = 1
            elif dirName in ['DME']:
                label = 2
            elif dirName in ['DRUSEN']:
                label = 3

            for fileName in tqdm(os.listdir(directory + dirName)):
                imageFile = cv2.imread(directory + dirName + '/' + fileName)
                if imageFile is not None:
                    imageFile = skimage.transform.resize(imageFile, (imageSize, imageSize, 3))
                    imageArray = np.asarray(imageFile)
                    x.append(imageArray)
                    y.append(label)
    
    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def load_data():

    print('Loading data...')
    x, y = extract_data(test_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test_samples')

    raw_y_train = y_train
    raw_y_test = y_test

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), (raw_y_train, raw_y_test)