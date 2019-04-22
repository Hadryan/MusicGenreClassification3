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


<<<<<<< HEAD:scripts/crnn/genre_rec_CRNN.py
def GenreRecCRNN(weights='msd', input_tensor=None, n_classes=10):
    '''Creates CRNN model for Music Genre Recognition
    
    inputs:
        weights (str): if None (random initialization)
=======
# def MusicTaggerCRNN(weights='msd', input_tensor=None):
#     '''Instantiate the MusicTaggerCRNN architecture,
#     optionally loading weights pre-trained
#     on Million Song Dataset. Note that when using TensorFlow,
#     for best performance you should set
#     `image_dim_ordering="tf"` in your Keras config
#     at ~/.keras/keras.json.

#     The model and the weights are compatible with both
#     TensorFlow and Theano. The dimension ordering
#     convention used by the model is the one
#     specified in your Keras config file.

#     For preparing mel-spectrogram input, see
#     `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
#     You will need to install [Librosa](http://librosa.github.io/librosa/)
#     to use it.

#     # Arguments
#         weights: one of `None` (random initialization)
#             or "msd" (pre-training on ImageNet).
#         input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#             to use as image input for the model.
#     # Returns
#         A Keras model instance.
#     '''

# #     if weights is None:
# #         return get_Model()
#     K.set_image_dim_ordering('th')
    
#     if weights not in {'msd', None}:
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization) or `msd` '
#                          '(pre-training on Million Song Dataset).')

#     # Determine proper input shape
#     if K.image_dim_ordering() == 'th':
#         input_shape = (1, 96, 1366)
#     else:
#         input_shape = (96, 1366, 1)

#     if input_tensor is None:
#         melgram_input = Input(shape=input_shape)
#     else:
#         melgram_input = Input(shape=input_tensor)

#     # Determine input axis
#     if K.image_dim_ordering() == 'th':
#         channel_axis = 1
#         freq_axis = 2
#         time_axis = 3
#     else:
#         channel_axis = 3
#         freq_axis = 1
#         time_axis = 2
    
#     # Input block
#     x = ZeroPadding2D(padding=(0, 37))(melgram_input)
#     x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

#     # RNN layer
#     # Conv block 1
#     x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
#     x = BatchNormalization(axis=channel_axis, name='bn1')(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
#     x = Dropout(0.1, name='dropout1')(x)

#     # Conv block 2
#     x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
#     x = BatchNormalization(axis=channel_axis, name='bn2')(x)
#     x = ELU()(x)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
#     x = Dropout(0.1, name='dropout2')(x)

#     # Conv block 3
#     x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
#     x = BatchNormalization(axis=channel_axis, name='bn3')(x)
#     x = ELU()(x)
#     x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
#     x = Dropout(0.1, name='dropout3')(x)

#     # Conv block 4
#     x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
#     x = BatchNormalization(axis=channel_axis, name='bn4')(x)
#     x = ELU()(x)
#     x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
#     x = Dropout(0.1, name='dropout4')(x)
    
    
#     # CNN to RNN
#     # x = Reshape(target_shape=((15, 128)), name='reshape')(x)  # (None, 32, 2048)
#     # x = Dense(128, activation='relu', kernel_initializer='normal', name='dense1')(x)  # (None, 32, 64)
#     if K.image_dim_ordering() == 'th':
#         x = Permute((3, 1, 2))(x)
#     x = Reshape((15, 128))(x)

    
#     # GRU block 1, 2, output
#     x = GRU(32, return_sequences=True, name='gru1')(x)
#     x = GRU(32, return_sequences=False, name='gru2')(x)
#     x = Dropout(0.3, name='final_drop')(x)

#     if weights is None:
#         # Create model
#         x = Dense(10, activation='sigmoid', name='output')(x)
#         model = Model(melgram_input, x)
#         return model
#     else:
#         # Load input
#         x = Dense(50, activation='sigmoid', name='output')(x)
#         if K.image_dim_ordering() == 'tf':
#             raise RuntimeError("Please set image_dim_ordering == 'th'."
#                                "You can set it at ~/.keras/keras.json")
#         # Create model
#         initial_model = Model(melgram_input, x)
        
#         initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % 'tensorflow',
#                            by_name=True)


#         # Eliminate last layer
#         pop_layer(initial_model)

#         # Add new Dense layer
#         last = initial_model.get_layer('final_drop')
#         # preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
#         preds = (Dense(8, activation='sigmoid', name='preds'))(last.output)
#         model = Model(initial_model.input, preds)

#         for layer in model.layers[:10]:
#             layer.trainable = False

#         return model

def MusicTaggerCRNN(weights='msd', input_tensor=None, n_classes=10):
    '''Instantiate the MusicTaggerCRNN architecture,
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
>>>>>>> 28acd9923f65f0392f8edab6b68e23e29e2835ff:scripts/crnn/tagger_net.py
            or "msd" (pre-training on ImageNet).
        input_tensor (tuple of ints): Keras tensor
            to use as image input for the model.
    outputs:
        A Keras model instance.
    '''

#     if weights is None:
#         return get_Model()
    K.set_image_dim_ordering('th')
    
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
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # RNN layer
    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)
    
    # model = Model(melgram_input, x)
    # model.summary()
    
    # CNN to RNN
    # x = Reshape(target_shape=((15, 128)), name='reshape')(x)  # (None, 32, 2048)
    # x = Dense(128, activation='relu', kernel_initializer='normal', name='dense1')(x)  # (None, 32, 64)
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    
    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
<<<<<<< HEAD:scripts/crnn/genre_rec_CRNN.py
        x = Dense(n_classes, activation='softmax', name='output')(x)
=======
        x = Dense(n_classes, activation='sigmoid', name='output')(x)
>>>>>>> 28acd9923f65f0392f8edab6b68e23e29e2835ff:scripts/crnn/tagger_net.py
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        # Theano dim ordering is reqeuired to use the pre-trained model
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
<<<<<<< HEAD:scripts/crnn/genre_rec_CRNN.py

        initial_model.load_weights('/home/stasdon/git/temp/musicgenrerecognition/scripts/crnn/weights/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
=======
        
        initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % 'tensorflow',
>>>>>>> 28acd9923f65f0392f8edab6b68e23e29e2835ff:scripts/crnn/tagger_net.py
                           by_name=True)
        
        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('final_drop')
<<<<<<< HEAD:scripts/crnn/genre_rec_CRNN.py
        preds = (Dense(n_classes, activation='softmax', name='preds'))(last.output)
=======
        preds = (Dense(n_classes, activation='sigmoid', name='preds'))(last.output)
>>>>>>> 28acd9923f65f0392f8edab6b68e23e29e2835ff:scripts/crnn/tagger_net.py
        model = Model(initial_model.input, preds)

        for layer in model.layers[:10]:
            layer.trainable = False

        return model