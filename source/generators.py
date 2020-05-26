import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from utils import get_data_dict, get_val_list, to_melspec, pad_with_zeros
import pandas as pd
import librosa


def image_generator(**kwargs):

    path = kwargs['path']
    validation_list = None if 'validation_list' not in kwargs else get_val_list(kwargs['validation_list'])
    validation_list = list(map(lambda x:path+'\\'+x, validation_list))

    words_dict = get_data_dict(path)
    del words_dict['_background_noise_']

    amount_dict = {k: len(v) for k, v in words_dict.items()}
    count_dict = {k: 0 for k in amount_dict}

    labels = {k: [0 for i in range(len(count_dict))] for k in count_dict}
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for k, i in words_dict.items():
            if count_dict[k] == amount_dict[k]:
                count_dict[k] = 0
            if validation_list is None or i[count_dict[k]] not in validation_list:
                yield img_to_array(load_img(i[count_dict[k]], target_size=(64, 64))), np.array(labels[k])
            count_dict[k] += 1


def validation_image_generator(name, path, validation_list):

    validation_list = get_val_list(validation_list)
    labels = {k: [0 for i in range(len(os.listdir(path)) - 1)] for k in os.listdir(path)}
    del labels['_background_noise_']
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for i in validation_list:
            yield img_to_array(load_img(path + "\\" + i, target_size=(64, 64))), np.array(labels[i[:i.index("\\")]])



def generator_nn(name, path, validation_list=None):
    
    df = pd.read_csv(path)

    df = df.sample(frac=1)
    
    labels = df['labels']
    
    df.drop(['labels'], inplace=True, axis=1)

    while True:
        for i in range(df.shape[0]):
            yield df.iloc[i].to_numpy(), list(map(lambda x: int(x),labels.iloc[i].strip('[]').replace(',', '').split()))


def generator_cnn_lstm(**kwargs):

    path = kwargs['path']
    validation_list = None if 'validation_list' not in kwargs else get_val_list(kwargs['validation_list'], 'wav')
    validation_list = list(map(lambda x:path+'\\'+x, validation_list))

    words_dict = get_data_dict(path)
    del words_dict['_background_noise_']

    amount_dict = {k: len(v) for k, v in words_dict.items()}
    count_dict = {k: 0 for k in amount_dict}

    labels = {k: [0 for i in range(len(count_dict))] for k in count_dict}
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for k, i in words_dict.items():
            if count_dict[k] == amount_dict[k]:
                count_dict[k] = 0
            if validation_list is None or i[count_dict[k]] not in validation_list:
                yield to_melspec(i[count_dict[k]]).T, np.array(labels[k])
            count_dict[k] += 1


def validation_generator_cnn_lstm(name, path, validation_list):

    validation_list = get_val_list(validation_list, 'wav')
    labels = {k: [0 for i in range(len(os.listdir(path)) - 1)] for k in os.listdir(path)}
    del labels['_background_noise_']
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1
    while True:
        for i in validation_list:
            yield to_melspec(path + "\\" + i).T, np.array(labels[i[:i.index('\\')]])

def generator_lstm(**kwargs):

    path = kwargs['path']
    validation_list = None if 'validation_list' not in kwargs else get_val_list(kwargs['validation_list'], 'wav')
    validation_list = list(map(lambda x:path+'\\'+x, validation_list))

    words_dict = get_data_dict(path)
    del words_dict['_background_noise_']

    amount_dict = {k: len(v) for k, v in words_dict.items()}
    count_dict = {k: 0 for k in amount_dict}

    labels = {k: [0 for i in range(len(count_dict))] for k in count_dict}
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for k, i in words_dict.items():
            if count_dict[k] == amount_dict[k]:
                count_dict[k] = 0
            if validation_list is None or i[count_dict[k]] not in validation_list:
                yield np.expand_dims(pad_with_zeros(librosa.load(i[count_dict[k]], sr=1000)[0], 1000), 1), np.array(labels[k])
            count_dict[k] += 1


def validation_lstm_generator(name, path, validation_list):

    validation_list = get_val_list(validation_list)
    labels = {k: [0 for i in range(len(os.listdir(path)) - 1)] for k in os.listdir(path)}
    del labels['_background_noise_']
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for i in validation_list:
            yield np.expand_dims(pad_with_zeros(librosa.load(path + "\\" + i, sr=8000)[0], 8000), 1), np.array(labels[i[:i.index("\\")]])