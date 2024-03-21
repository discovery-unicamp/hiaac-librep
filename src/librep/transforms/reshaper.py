import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike


class SimpleReshaper(Transform):
    """Reshape a single sample into n non-overlapping subsamples.

    Parameters
    ----------
    subsamples_number : int
        The number of subsamples the original data will be divided into.
    Returns
        -------
        ArrayLike
            The resampled samples with shape: (n_samples, new_sample_size, ).

        """

    def __init__(self, subsamples_number: int):
        self.subsamples_number = subsamples_number
    
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Reshape signal samples.

        Parameters
        ----------
        X : ArrayLike
            The signal samples with shape: (n_samples, n_features, )

        Returns
        -------
        ArrayLike
            The reshaped samples with shape: (n_samples, subsamples_number, -1).

        """
        return np.reshape(X, (X.shape[0], self.subsamples_number, -1))