from keras.layers import Input, SeparableConv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model


def build_cnn_v1(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)

    conv1 = SeparableConv2D(32, 3, padding="same", activation="relu")(inputs)
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
