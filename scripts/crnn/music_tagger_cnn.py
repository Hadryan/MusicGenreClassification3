# -*- coding: utf-8 -*-
'''MusicTaggerCNN model for Keras.

# Reference:

- [Automatic tagging using deep convolutional neural networks](https://arxiv.org/abs/1606.00298)
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

'''
from __future__ import print_function
from __future__ import absolute_import

import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, BatchNormalization
from keras.layers import MaxPooling2D, ZeroPadding2D, Conv2D, Flatten
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense

K.set_image_dim_ordering('th')

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def MusicTaggerCNN(weights='msd', input_tensor=None, n_classes=10):
    '''Instantiate the MusicTaggerCNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 256-dim features.


    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

        # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Conv2D(32, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Conv2D(192, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

    # Conv block 5
    x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten(name='Flatten_1')(x)

    if weights is None:
        # Create model
        x = Dense(n_classes, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        print(K.image_dim_ordering())
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        
        # initial_model.load_weights('weights/music_tagger_cnn_weights_%s.h5' % 'theano',
        #                            by_name=True)
        initial_model.load_weights('weights/music_tagger_cnn_weights_%s.h5' % 'tensorflow',
                                   by_name=True)

        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('Flatten_1')
        preds = (Dense(n_classes, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        for layer in model.layers[:-6]:
            layer.trainable = False

        return model
