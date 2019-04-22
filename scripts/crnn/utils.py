import os
import time
import h5py
import sys
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle
import librosa
from math import floor

sys.path.append('/home/stasdon/git/musicgenrerecognition/scripts/')
import config as cfg


def compute_melgram(audio_path):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..


    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def load_gt(path):
    data = pd.read_csv(path)
    labels_df = pd.read_csv(cfg.DATASET_PATH + 'labels.csv')
    le = LabelEncoder()
    le.fit_transform(labels_df.genre.unique())
    genres_map = dict(zip(le.classes_, le.transform(le.classes_)))

    labels = data.genre.apply(lambda x: genres_map.get(x))
    return labels

<<<<<<< HEAD
    
def extract_melgrams(data_path):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    data = pd.read_csv(data_path)
    song_paths = data.track_id
<<<<<<< HEAD
    
    labels_df = pd.read_csv(cfg.DATASET_PATH + 'labels.csv')
    le = LabelEncoder()
    le.fit_transform(labels_df.genre.unique())
    genres_map = dict(zip(le.classes_, le.transform(le.classes_)))
    
=======
=======

def plot_confusion_matrix(cnf_matrix, classes, title):

    cnfm_suma=cnf_matrix.sum(1)
    cnfm_suma_matrix = np.repeat(cnfm_suma[:,None],cnf_matrix.shape[1],axis=1)

    cnf_matrix=10000*cnf_matrix/cnfm_suma_matrix
    cnf_matrix=cnf_matrix/(100*1.0)
    print(cnf_matrix)

    #print map(truediv,cnf_matrix, cnfm_suma_matrix)

    fig=plt.figure()
    cmap=plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #plt.show()
    fig.savefig(title)


# Melgram computation
def extract_melgrams(data_path):#, MULTIFRAMES, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    data = pd.read_csv(data_path)
    song_paths = data.track_id

    with open(cfg.DATASET_PATH + 'genres_map.pckl', 'rb') as f:
        genres_map = pickle.load(f)
        
    labels = data.genre.apply(lambda x: genres_map.get(x)).values
    
    song_labels = []
    for song_ind, song_path in tqdm(enumerate(song_paths), total=len(data)):
        try:
            melgram = compute_melgram(cfg.AUDIO_DIR + song_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
            song_labels.append(labels[song_ind])
        except Exception as ex:
            print(song_path, ex)

    return melgrams, song_labelsf
