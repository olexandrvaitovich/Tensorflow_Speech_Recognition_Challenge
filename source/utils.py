import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


def get_data_dict(path):
    return {i:[path+"\\"+i+"\\"+j for j in os.listdir(path+"\\"+i)]\
            for i in [k for k in os.listdir(path)]}


def pad_with_zeros(data, length=22050):
    return librosa.util.fix_length(data, length)


def mel_spec_to_img(sample_path, file_path):
    X, sr = librosa.load(sample_path, res_type='kaiser_fast')
    X = pad_with_zeros(X)
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='linear')
    plt.savefig(file_path, dpi=500, bbox_inches='tight',pad_inches=0)
    plt.close()


def get_val_list(VAL_LIST_PATH, ending='png'):
    with open(VAL_LIST_PATH, 'r') as f:
        return list(map(lambda x: x.replace("/", "\\").replace("wav\n", ending), f.readlines()))




def make_dir(path):
    from datetime import datetime
    now = datetime.now()
    folder_path = path+f"\\{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"
    os.mkdir(folder_path)
    return folder_path


def get_features(file_path):
    
    X, sr = librosa.load(file_path, res_type='kaiser_fast') 

    stft = np.abs(librosa.stft(X))
    
    return np.concatenate([np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1), 
                           np.mean(librosa.feature.melspectrogram(X, sr=sr), axis=1), 
                           np.mean(librosa.feature.mfcc(y=X, sr=sr), axis=1),
                           np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1), 
                           np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr), axis=1),
                           [np.mean(librosa.feature.zero_crossing_rate(X))]])


def to_melspec(path, n_mels=128):

    X, sr = librosa.load(path, res_type='kaiser_fast')
    X = pad_with_zeros(X)
    mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=n_mels)

    return librosa.power_to_db(mel_spectrogram, ref=np.max)
