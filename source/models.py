from keras.layers import Input, SeparableConv2D, MaxPooling2D, Dense, Flatten, SeparableConv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization
from keras.models import Model


def build_cnn_v1(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    conv1 = SeparableConv2D(32, 3, padding="same", activation="relu")(bn)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = SeparableConv2D(64, 3, padding="same", activation="relu")(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = SeparableConv2D(128, 3, padding="same", activation="relu")(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = SeparableConv2D(256, 3, padding="same", activation="relu")(pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)

    flatten = Flatten()(pool4)

    dense1 = Dense(100, activation='relu')(flatten)
    dense2 = Dense(60, activation='relu')(dense1)

    output = Dense(30, activation='softmax')(dense2)

    return Model(inputs=[inputs], outputs=[output])


def build_nn_v1(input_shape=(174,)):

    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    dense1 = Dense(256, activation='relu')(bn)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(512, activation='relu')(drop1)
    drop2 = Dropout(0.2)(dense2)

    dense3 = Dense(512, activation='relu')(drop2)
    drop3 = Dropout(0.2)(dense3)

    dense4 = Dense(256, activation='relu')(drop3)
    drop4 = Dropout(0.2)(dense4)

    output = Dense(30, activation='softmax')(drop4)

    return Model(inputs=[inputs], outputs=[output])


def build_nn_v2(input_shape=(174,)):

    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    dense1 = Dense(128, activation='relu')(bn)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(128, activation='relu')(drop1)
    drop2 = Dropout(0.2)(dense2)

    dense3 = Dense(256, activation='relu')(drop2)
    drop3 = Dropout(0.2)(dense3)

    dense4 = Dense(512, activation='relu')(drop3)
    drop4 = Dropout(0.2)(dense4)

    dense5 = Dense(256, activation='relu')(drop4)
    drop5 = Dropout(0.2)(dense5)

    dense6 = Dense(128, activation='relu')(drop5)
    drop6 = Dropout(0.2)(dense6)

    dense7 = Dense(128, activation='relu')(drop6)
    drop7 = Dropout(0.2)(dense7)

    output = Dense(30, activation='softmax')(drop7)

    return Model(inputs=[inputs], outputs=[output])


def build_cnn_lstm_v1(input_shape=(44, 128)):

    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    conv1 = SeparableConv1D(32, 3, activation='relu', padding='same')(bn)
    conv2 = SeparableConv1D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D()(conv2)
    lstm1 = LSTM(64, return_sequences=True)(pool1)

    conv3 = SeparableConv1D(64, 3, activation='relu', padding='same')(lstm1)
    conv4 = SeparableConv1D(64, 3, activation='relu', padding='same')(conv3)
    pool2 = MaxPooling1D()(conv4)
    lstm2 = LSTM(128, return_sequences=True)(pool2)

    conv5 = SeparableConv1D(128, 3, activation='relu', padding='same')(lstm2)
    conv6 = SeparableConv1D(128, 3, activation='relu', padding='same')(conv5)
    pool3 = MaxPooling1D()(conv6)

    flatten = Flatten()(pool3)


    dense1 = Dense(128, activation='relu')(lstm3)

    output = Dense(30, activation='softmax')(dense1)

    return Model(inputs=[inputs], outputs=[output])



def build_cnn1d_v1(input_shape=(44, 128)):

    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    conv1 = SeparableConv1D(32, 5, activation='relu', padding='same')(bn)
    conv2 = SeparableConv1D(32, 5, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D()(conv2)

    conv3 = SeparableConv1D(64, 3, activation='relu', padding='same')(pool1)
    conv4 = SeparableConv1D(64, 3, activation='relu', padding='same')(conv3)
    pool2 = MaxPooling1D()(conv4)

    conv5 = SeparableConv1D(128, 3, activation='relu', padding='same')(pool2)
    conv6 = SeparableConv1D(128, 3, activation='relu', padding='same')(conv5)
    pool3 = MaxPooling1D()(conv6)

    flatten = Flatten()(pool3)

    dense1 = Dense(256, activation='relu')(flatten)

    output = Dense(30, activation='softmax')(dense1)

    return Model(inputs=[inputs], outputs=[output])



def build_lstm_v1(input_shape=(8000, 1)):

    inputs = Input(shape=input_shape)

    bn = BatchNormalization()(inputs)

    lstm1 = LSTM(128)(bn)

    output = Dense(30, activation='softmax')(lstm1)

    return Model(inputs=[inputs], outputs=[output])

