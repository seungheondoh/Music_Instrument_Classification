import sys
import os
import numpy as np
import librosa
import pandas as pd
import sklearn

from feature_engineering import *
from config import CreateDataset

data_path = CreateDataset.data_path
csv_name = CreateDataset.Name

# file load
def get_sampels(data_set='train'):
    audios = []
    labels = []
    path_of_audios = librosa.util.find_files(data_path + data_set)
    for audio in path_of_audios:
        labels.append(audio.split('train/')[1].split('_')[0])
        y, sr = librosa.load(audio, sr=22050, duration=4.0)
        audios.append(y)
    audios_numpy = np.array(audios)
    return audios_numpy, labels

def main():
    is_created = False
    audios_numpy, labels = get_sampels(data_set='train')
    for samples in audios_numpy:
        row = extract_feature(samples)
        if not is_created:
            dataset_numpy = np.array(row)
            is_created = True
        elif is_created:
            dataset_numpy = np.vstack((dataset_numpy, row))

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)

    dataset_pandas = pd.DataFrame(dataset_numpy)
    dataset_pandas["instruments"] = labels
    dataset_pandas.to_csv(csv_name, index=False)

if __name__ == '__main__':
    main()



