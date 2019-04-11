from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import 
from keras.layers.normalization import BatchNormalization
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


def MusicTaggerCRNN(weights='msd', input_tensor=None):
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
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
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
    x = ZeroPadding2D(padding=(0, 37), data_format='channels_first')(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # RNN layer
    # Conv block 1
    x = Conv2D(64, (3, 3), border_mode='same', name='conv1', data_format='channels_first')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', dim_ordering="th")(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), border_mode='same', name='conv2', data_format='channels_first')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2', dim_ordering="th")(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), border_mode='same', name='conv3', data_format='channels_first')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', dim_ordering="th")(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), border_mode='same', name='conv4', data_format='channels_first')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', dim_ordering="th")(x)
    x = Dropout(0.1, name='dropout4')(x)
    
    model = Model(melgram_input, x)
    model.summary()
    
    print(x.shape)
    # CNN to RNN
    x = Reshape(target_shape=((15, 128)), name='reshape')(x)  # (None, 32, 2048)
    x = Dense(64, activation='relu', init='normal', name='dense1')(x)  # (None, 32, 64)


    # reshaping
    # if K.image_dim_ordering() == 'th':
    #     x = Permute((3, 1, 2))(x)
    # x = Reshape((15, 128))(x)
#    x = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
#    x = Dense(64, activation='relu', init='normal', name='dense1')(inner)  # (None, 32, 
    
    print(x.shape)
    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
        x = Dense(8, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)

        initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)


        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('final_drop')
        preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
#        preds = (Dense(8, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        for layer in model.layers[:-6]:
            layer.trainable = False

        return model


# def get_Model():
#     input_shape = (1, 96, 1366)

#     # Make Networkw
#     inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

#     # Convolution layer (VGG)
#     inner = Conv2D(64, (3, 3), padding='same', name='conv1', init='normal')(inputs)  # (None, 128, 64, 64)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max1', dim_ordering="th")(inner)  # (None,64, 32, 64)

#     inner = Conv2D(128, (3, 3), padding='same', name='conv2', init='normal')(inner)  # (None, 64, 32, 128)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max2', dim_ordering="th")(inner)  # (None, 32, 16, 128)

#     inner = Conv2D(256, (3, 3), padding='same', name='conv3', init='normal')(inner)  # (None, 32, 16, 256)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(256, (3, 3), padding='same', name='conv4', init='normal')(inner)  # (None, 32, 16, 256)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(1, 2), name='max3', dim_ordering="th")(inner)  # (None, 32, 8, 256)

#     inner = Conv2D(512, (3, 3), padding='same', name='conv5', init='normal')(inner)  # (None, 32, 8, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(1, 2), name='max4', dim_ordering="th")(inner)  # (None, 32, 4, 512)

#     inner = Conv2D(512, (2, 2), padding='same', init='normal', name='con7')(inner)  # (None, 32, 4, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)

#     print(inner.shape)
#     # CNN to RNN
#     inner = Reshape(target_shape=((15, 128)), name='reshape')(inner)  # (None, 32, 2048)
#     inner = Dense(64, activation='relu', init='normal', name='dense1')(inner)  # (None, 32, 64)

#     # RNN layer
#     gru_1 = GRU(256, return_sequences=True, init='normal', name='gru1')(inner)  # (None, 32, 512)
#     gru_1b = GRU(256, return_sequences=True, go_backwards=True, init='normal', name='gru1_b')(inner)
#     reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

#     gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
#     gru1_merged = BatchNormalization()(gru1_merged)
    
#     gru_2 = GRU(256, return_sequences=True, init='normal', name='gru2')(gru1_merged)
#     gru_2b = GRU(256, return_sequences=True, go_backwards=True, init='normal', name='gru2_b')(gru1_merged)
#     reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

#     gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
#     gru2_merged = BatchNormalization()(gru2_merged)

#     # transforms RNN output to character activations:
#     inner = Dense(10, init='normal',name='dense2')(gru2_merged) #(None, 32, 63)
#     y_pred = Activation('softmax', name='softmax')(inner)

#     return Model(inputs=[inputs], outputs=y_pred)