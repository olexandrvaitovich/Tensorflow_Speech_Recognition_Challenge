import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


def get_data_dict(path):
    return {i:[path+"\\"+i+"\\"+j for j in os.listdir(path+"\\"+i)] for i in [i for i in os.listdir(path)]}


def mel_spec_to_img(sample_path, file_path):
    X, sr = librosa.load(sample_path, res_type='kaiser_fast')
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='linear')
    plt.savefig(file_path, dpi=500, bbox_inches='tight',pad_inches=0)
    plt.close()


def get_val_list(VAL_LIST_PATH):
    with open(VAL_LIST_PATH, 'r') as f:
        return list(map(lambda x: x.replace("/", "\\").replace("wav\n", "png"), f.readlines()))


def make_dir(path):
    from datetime import datetime
    now = datetime.now()
    folder_path = path+f"\\{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"
    os.mkdir(folder_path)
    return folder_path
