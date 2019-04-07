import sys
import os
import numpy as np
import pandas as pd
import librosa
from sklearn import svm
import joblib

from config import Model, CreateDataset

def main():
    # Load data
    data_set = pd.read_csv(CreateDataset.Name, index_col=False)
    data_set = np.array(data_set)

    # Cacluate Shape
    row, col = data_set.shape
    # Get X and Y
    x = data_set[:, :col-1]
    y = data_set[:, col-1]

    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=0.02, kernel='linear',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    clf.fit(x, y)

    joblib.dump(clf, Model.NAME)

if __name__ == '__main__':
    main()


