import os

DATASET_PATH = '/home/stasdon/git/musicgenrerecognition/data/GTZAN/'
AUDIO_DIR = DATASET_PATH + 'data/'
EXP_NAME = DATASET_PATH.split(os.sep)[-2]
MODEL_NAME = '16_04_cnn_net_transfer_' + EXP_NAME

LABELS_PATH = DATASET_PATH + 'labels.csv'
# TRAIN_PATH = DATASET_PATH + 'train.csv'
# TEST_PATH = DATASET_PATH + 'test.csv'

LOAD_WEIGHTS = ''
BATCH_SIZE = 128
NB_EPOCH = 40