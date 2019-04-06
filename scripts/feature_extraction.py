#!/usr/bin/env python3

import os
import multiprocessing
import warnings

from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd
import librosa
import argparse

import utils
import config as cfg

SAVE_DIR = os.sep.join(cfg.AUDIO_DIR.split(os.sep)[:-1])


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid):
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)
    tid = "{:06d}".format(tid)
    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        # filepath = os.path.join(cfg.AUDIO_DIR, tid[:3], tid + '.mp3')
        filepath = os.path.join(cfg.AUDIO_DIR, tid + '.mp3')
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rmse(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(filepath, repr(e)))

    return features


def save(features, ndigits):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)


    features.to_csv(os.path.join(cfg.AUDIO_DIR, 'features.csv'), float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = utils.load(os.path.join(cfg.AUDIO_DIR, 'features.csv'))
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    labels_df = pd.read_csv(os.path.join(cfg.AUDIO_DIR, 'labels.csv'), index_col='track_id')
    tracks_df = utils.load(os.path.join(cfg.AUDIO_DIR, 'tracks.csv'))

    features = pd.DataFrame(index=labels_df.index,
                            columns=columns(), dtype=np.float32)

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * 6)


    # tids = labels_df.index
    # pool = multiprocessing.Pool(nb_workers)
    # it = pool.imap_unordered(compute_features, tids)
    #
    # for i, row in enumerate(tqdm(it, total=len(tids))):
    #     features.loc[row.name] = row
    #
    #     if i % 100 == 0:
    #         save(features, 10)
    #
    # Longest is ~11,000 seconds. Limit processes to avoid memory errors.
    table = ((5000, 1), (3000, 3), (2000, 5), (1000, 10), (0, nb_workers))
    for duration, nb_workers in table:
        print('Working with {} processes.'.format(nb_workers))

        tids = tracks_df[tracks_df['track', 'duration'] >= duration].index
        tracks_df.drop(tids, axis=0, inplace=True)

        pool = multiprocessing.Pool(nb_workers)
        it = pool.imap_unordered(compute_features, tids)

        for i, row in enumerate(tqdm(it, total=len(tids))):
            features.loc[row.name] = row

            if i % 100 == 0:
                save(features, 10)

    save(features, 10)
    test(features, 10)    # print('Hello')