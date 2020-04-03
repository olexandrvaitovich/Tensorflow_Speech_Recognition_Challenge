import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from utils import get_data_dict, get_val_list


def image_generator(**kwargs):

    path = kwargs['path']
    validation_list = None if 'validation_list' not in kwargs else get_val_list(kwargs['validation_list'])
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
            if validation_list is None or i[count_dict[k]][i[count_dict[k]][::-1].index('\\'):] not in validation_list:
                yield np.expand_dims(img_to_array(load_img(i[count_dict[k]], target_size=(64, 64))), 0), np.expand_dims(
                    np.array(labels[k]), 0)
            count_dict[k] += 1


def validation_image_generator(path, validation_list):

    validation_list = get_val_list(validation_list)
    labels = {k: [0 for i in range(len(os.listdir(path)) - 1)] for k in os.listdir(path)}
    del labels['_background_noise_']
    count = 0
    for i in labels:
        labels[i][count] = 1
        count += 1

    while True:
        for i in validation_list:
            yield np.expand_dims(img_to_array(load_img(path + "\\" + i, target_size=(64, 64))), 0), np.expand_dims(
                np.array(labels[i[:i.index("\\")]]), 0)


