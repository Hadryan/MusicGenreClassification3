import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import np_utils
import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, BatchNormalization
from keras.layers import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
import logging

from utils import extract_melgrams
sys.path.append('../')
import config as cfg


MODEL_PATH = "models_trained/" + cfg.MODEL_NAME + "/"
WEIGHTS_PATH = MODEL_PATH + "/weights/"

# Create directories for the models & weights
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    print('Path created: ', MODEL_PATH)

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
    print('Path created: ', WEIGHTS_PATH)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


class Model:
    def __init__(self, ModelCreator, weights='msd'):
        self.get_nb_classes()
        self.load_data()
        self.model = ModelCreator(weights, self.X_train.shape[1:], self.nb_classes)
        self.prepare_model()
    
    def load_data(self):
        print('Extracting features for train set:')
        
        
        if os.path.exists(cfg.DATASET_PATH + 'features.pckl'):
            with open(cfg.DATASET_PATH + 'features.pckl', 'rb') as f:
                X = pickle.load(f)
            with open(cfg.DATASET_PATH + 'targets.pckl', 'rb') as f:
                y = pickle.load(f)
        else:
            X, y = extract_melgrams(cfg.LABELS_PATH)
            pickle.dump(X, open(cfg.DATASET_PATH + 'features.pckl', "wb"), protocol=4)
            pickle.dump(y, open(cfg.DATASET_PATH + 'targets.pckl', "wb"), protocol=4)
            
        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=23)
        
        print('X_train shape:', self.X_train.shape)
        
        self.X_val, self.X_test, y_val, y_test = train_test_split(self.X_test, y_test, stratify=y_test, test_size=0.5, random_state=23)
        print('X_val shape:', self.X_val.shape)
        print('X_test shape:', self.X_test.shape)
        
        self.Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        self.Y_val = np_utils.to_categorical(y_val, self.nb_classes)
        self.Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        
        
    def get_nb_classes(self):
        with open(cfg.DATASET_PATH + 'genres_map.pckl', 'rb') as f:
            self.genres_map = pickle.load(f)
        self.nb_classes = len(self.genres_map)
        
    def prepare_model(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        if cfg.LOAD_WEIGHTS:
            print('Loading weights...')
            self.model.load_weights(cfg.LOAD_WEIGHTS)
        self.model.summary()
        
    def fit(self):
        filepath= WEIGHTS_PATH + "{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint, TestCallback((self.X_test, self.Y_test))]
        self.model.fit(self.X_train, self.Y_train, nb_epoch=cfg.NB_EPOCH, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=1, validation_data=(self.X_val, self.Y_val))
        
        
    def evaluate(self):
        return self.model.evaluate(self.X_test, self.Y_test, batch_size=cfg.BATCH_SIZE, verbose=1)
    