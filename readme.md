# Musical Instrument Classification

The most basic element that composes a music is an instrument. The combination of instruments creates new music and is delivered to the user. Musical instruments are given their own characteristics according to their manufacturing methods and materials. Understanding the unique characteristics of these instruments is expected to make a significant contribution to music classification and generation models.
In this report, The automatic classification of musical instruments, using NSynth dataset, which contain 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. We use three feature group for representing timbre features, pitch/harmony features, rhythm features. Experiments use old Machine-Learning algorithm and 5fold-cross validation. The main points of this report are using small data set(Train 1200, Test 200) and focusing on relationship between 3 feature group and Machine-Learning algorithm. Using the proposed feature group, classification of 92.5% for 10 class of musical instruments.

### Dataset
We use a subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) dataset which is a large collection of musical instrument tones from the Google Magenta project. The subset has 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. The data-set are all .wav file. The files that stored as 16000Hz, 16-bit, 4sec, mono audio file. 

<img src="/img/visualization.png">

### Install package via pip install
```
$ sudo pip3 install numpy scipy pandas scikit-learn librosa matplotlib
```

For use this repository, you need to install this requirements,
- python3.6
- IPython
- Numpy
- Scipy
- Pandas
- Librosa
- Scikit-learn
- Matplotlib

### Learning code
```
$ python main.py
```

After you build model and make a .pkl file you just change test set and   

```
$ python test.py
```

### result

Classification accuracy varied depending on the nature of the machine learning algorithm and feature group. The final best result was the Timber + Rhythm feature with 92.5 accuracy and SVM model. The SVM model performed better than the other models overall. Timber features also showed better results than other features.

hypotheses set | SVM | Softmax | GaussianNB | KNN | DT | RF
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Timber | 91.0 | 88.0 | 77.5 | 84.5 | 70.5 | 88.0
Pitch/harmony | 63.5 | 36.5 | 42 | 55.0 | 44.5 | 55
Rhythm | 15.5 | 14.5 | 14.0 | 8.5 | 6.5 | 6.5
Timber + Pitch/harmony | 90.5 | 88.0 | 30 | 71.0 | 63.5 | 89.0
Timber + Rhythm | __92.5__ | 88.0 | 73.0 | 85.5 | 61.5 | 88.5
Pitch/harmony + Rhythm | 64 | 40.5 | 30.0 | 56.0 | 20.0 | 45.5 
Timber + Rhythm + Pitch/harmony | 91.5 | 87.0 | 77.5 | 73.0 | 62.5 | 90.5
