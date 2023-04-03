import numpy as np
from scipy import signal

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class Spectrogram(Transform):

    def __init__(self,
                 fs: float = 100,
                 segment_size: int = 50,
                 overlap: int = 30,
                 reshape: bool = True):
        self.fs = fs
        self.segment_size = segment_size
        self.overlap = overlap
        self.reshape = True

    # TODO
    def transform(self, X: ArrayLike):
        f, t, Sxx = signal.spectrogram(X,
                                       fs=self.fs,
                                       nperseg=self.segment_size,
                                       noverlap=self.overlap)
        if self.reshape:
            return Sxx.ravel()
        else:
            return Sxx