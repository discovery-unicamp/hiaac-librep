from librep.base.transform import Transform
import numpy as np
from scipy import signal

class Spectrogram(Transform):
    def __init__(self, fs: int = 20, nperseg=20, nfft=None, window=("tukey", 0.25)):
        self.fs = fs
        self.nperseg = nperseg
        self.nfft = nfft
        self.window = window

    def transform(self, X):
        datas = []
        for data in X:
            _, _, Sxx = signal.spectrogram(data, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft, window=self.window)
            datas.append(Sxx.ravel())
        return np.array(datas)
