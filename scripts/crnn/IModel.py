import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import np_utils
import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, BatchNormalization
from keras.layers import MaxPooling2D, ZeroPadding2D, Conv2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import 
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
import logging

from utils import extract_melgrams
from music_tagger_cnn import MusicTaggerCNN
from tagger_net import MusicTaggerCRNN
sys.path.append('../')
import config as cfg


LOAD_WEIGHTS = 0
EXP_NAME = cfg.DATASET_PATH.split(os.sep)[-2]
MODEL_NAME = "14_04_crnn_net_adam_scratch_" + EXP_NAME 
MODEL_PATH = "models_trained/" + MODEL_NAME + "/"
WEIGHTS_PATH = "models_trained/" + MODEL_NAME + "/weights/"

# Create directories for the models & weights
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    print('Path created: ', MODEL_PATH)

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
    print('Path created: ', WEIGHTS_PATH)

class Evaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()

        self.X_test, self.Y_test = validation_data

    def on_epoch_end(self, epoch, logs={}):
        score = self.model.evaluate(self.X_test, self.Y_test, batch_size=cfg.BATCH_SIZE, verbose=0)
        logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score[0]))


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
        
#         if os.path.exists(cfg.DATASET_PATH + 'train.pckl'):
#             with open(cfg.DATASET_PATH + 'train.pckl', 'rb') as f:
#                 self.X_train = pickle.load(f)
#             with open(cfg.DATASET_PATH + 'train_gt.pckl', 'rb') as f:
#                 y_train = pickle.load(f)
#         else:
#             self.X_train, y_train = extract_melgrams(cfg.TRAIN_PATH, MULTIFRAMES, process_all_song=False, num_songs_genre=20)
#             pickle.dump(self.X_train, open(cfg.DATASET_PATH + 'train.pckl', "wb"), protocol=4)
#             pickle.dump(y_train, open(cfg.DATASET_PATH + 'train_gt.pckl', "wb"), protocol=4)
        print('X_train shape:', self.X_train.shape)

#         print('Extracting features for test set:')
#         if os.path.exists(cfg.DATASET_PATH + 'test.pckl'):
#             with open(cfg.DATASET_PATH + 'test.pckl', 'rb') as f:
#                 self.X_test = pickle.load(f)

#             with open(cfg.DATASET_PATH + 'test_gt.pckl', 'rb') as f:
#                 y_test = pickle.load(f)
#         else:
#             self.X_test, y_test = extract_melgrams(cfg.TEST_PATH, MULTIFRAMES, process_all_song=False, num_songs_genre=10)
#             pickle.dump(self.X_test, open(cfg.DATASET_PATH + 'test.pckl', "wb"), protocol=4)
#             pickle.dump(y_test, open(cfg.DATASET_PATH + 'test_gt.pckl', "wb"), protocol=4)
        
        # self.X_val = self.X_test[:len(self.X_test)//2]
        
        self.X_val, self.X_test, y_val, y_test = train_test_split(self.X_test, y_test, stratify=y_test, test_size=0.5, random_state=23)
        print('X_val shape:', self.X_val.shape)
        
        # self.X_test = self.X_test[len(self.X_test)//2:]
        print('X_test shape:', self.X_test.shape)
        
#         y_train = np.array(y_train)
#         y_test = np.array(y_test)
#         y_val = y_test[:len(y_test)//2]
#         y_test = y_test[len(y_test)//2:]

        self.Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        self.Y_val = np_utils.to_categorical(y_val, self.nb_classes)
        self.Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        
        
    def get_nb_classes(self):
        labels_df = pd.read_csv(cfg.DATASET_PATH + 'labels.csv')
        le = LabelEncoder()
        le.fit_transform(labels_df.genre.unique())

        tags = le.classes_
        tags = np.array(tags)

        self.nb_classes = len(tags)
        self.genres_map = dict(zip(le.classes_, le.transform(le.classes_)))
        
    def prepare_model(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        if LOAD_WEIGHTS:
            print('Loading weights...')
            self.model.load_weights(WEIGHTS_PATH + MODEL_NAME + 'epoch{}.h5'.format(LOAD_WEIGHTS))
        self.model.summary()
        
    def fit(self):
        filepath= WEIGHTS_PATH + MODEL_NAME + "_{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        evluation_callback = Evaluation(validation_data=(self.X_test, self.Y_test))
        callbacks_list = [checkpoint, evluation_callback]
        self.model.fit(self.X_train, self.Y_train, epochs=cfg.NB_EPOCH, batch_size=cfg.BATCH_SIZE, callbacks=callbacks_list, verbose=1, validation_data=(self.X_val, self.Y_val))
        
        
    def evaluate(self):
        return self.model.evaluate(self.X_test, self.Y_test, batch_size=cfg.BATCH_SIZE, verbose=1)
    