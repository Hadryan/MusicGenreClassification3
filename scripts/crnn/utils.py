import os
import time
import h5py
import sys
import librosa
import audio_processor as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

sys.path.append('/home/stasdon/git/musicgenrerecognition/scripts/')
import config as cfg


# Functions Definition
def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print(name, ':', '%5.3f  ' % score, '   ',)
    print()


def predict_label(preds):
    labels=preds.argsort()[::-1]
    return labels[0]


def load_gt(path):
    data = pd.read_csv(path)
    labels_df = pd.read_csv(cfg.DATASET_PATH + 'labels.csv')
    le = LabelEncoder()
    le.fit_transform(labels_df.genre.unique())
    genres_map = dict(zip(le.classes_, le.transform(le.classes_)))

    labels = data.genre.apply(lambda x: genres_map.get(x))
    return labels


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
def extract_melgrams(data_path, MULTIFRAMES, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    data = pd.read_csv(data_path)
    song_paths = data.track_id
    
    labels_df = pd.read_csv(cfg.DATASET_PATH + 'labels.csv')
    le = LabelEncoder()
    le.fit_transform(labels_df.genre.unique())
    genres_map = dict(zip(le.classes_, le.transform(le.classes_)))
    
    labels = data.genre.apply(lambda x: genres_map.get(x)).values
    
    song_labels = []
    for song_ind, song_path in tqdm(enumerate(song_paths), total=len(data)):
        try:
            melgram = ap.compute_melgram(cfg.AUDIO_DIR + song_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
            song_labels.append(labels[song_ind])
        except Exception as ex:
            print(song_path, ex)
    
    return melgrams, song_labels
