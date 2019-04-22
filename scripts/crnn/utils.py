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
from sklearn.preprocessing import LabelEncoder
import pickle
import librosa
from math import floor

sys.path.append('../')
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
    labels = data.genre.apply(lambda x: cfg.GENRES_MAP.get(x))
    return labels

    
def extract_melgrams(data_path):
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
    
    return melgrams, song_labels
