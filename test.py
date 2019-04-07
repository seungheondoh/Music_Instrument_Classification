import joblib
import sklearn
import librosa
import numpy as np

from feature_engineering import *
from config import *

PATH = librosa.util.find_files(Test_path.data_path)

def main():
    labels = []
    samples = []
    for p in PATH:
        labels.append(p.split('test/')[1].split('_')[0])
        sample, sr = librosa.load(p, sr=22050, duration=4.0)
        samples.append(sample)

    data = np.array([extract_feature(sample) for sample in samples])

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data)

    clf = joblib.load(Model.NAME)
    test_Y_hat = clf.predict(data)

    accuracy = np.sum((test_Y_hat == labels)) / 200.0 * 100.0

    print('test accuracy = ' + str(accuracy) + ' %')

if __name__ == '__main__':
    main()
