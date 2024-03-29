from librep.config.type_definitions import ArrayLike, PathLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
# Borrowed from Sklearn API


class Estimator(Parametrizable):
    """An object which manages the estimation and decoding of a model."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
        **estimator_params,
    ) -> "Estimator":
        """Fits the model function arround X. It takes some samples (X) and
        the respective labels (y) if the model is supervised.
        X_val and y_val are optional and may be used if needed.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole training dataset with shape:
            (n_samples, n_features).
        y : ArrayLike
            The respective train labels, with shape: (n_samples, ). This parameter is
            optional and may be used if needed. By default None.
        X_val : ArrayLike
            An array-like representing the whole validation dataset with shape:
            (k_samples, n_features). This parameter is optional and may be used
            if needed. By default None.
        y_val : ArrayLike
            The respective validation labels, with shape: (k_samples, ). This
            parameter is optional and may be used if needed. By default None.
        **estimator_params : Any
            Optional data-dependent parameters.

        Returns
        -------
        'Estimator'
            The estimator (self).

        """
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict or project the values of the samples (X).

        Parameters
        ----------
        X : ArrayLike
            An array-like of samples, with shape: (n_features, ).

        Returns
        -------
        ArrayLike
            An array-like describing the respective labels predicted for each
            samples.

        """

        raise NotImplementedError

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
        **estimator_params,
    ) -> ArrayLike:
        """Chain fit and predict methods, togheter. It fits the model and
        predict the samples of X.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole training dataset with shape:
            (n_samples, n_features).
        y : ArrayLike
            The respective training labels, with shape: (n_samples, ). This parameter is
            optional and may be used if needed. By default None.
        X_val : ArrayLike
            An array-like representing the whole validation dataset with shape:
            (k_samples, n_features). This parameter is optional and may be used
            if needed. By default None.
        y_val : ArrayLike
            The respective validation labels, with shape: (k_samples, ). This
            parameter is optional and may be used if needed. By default None.
        **estimator_params : type
            Optional data-dependent parameters.

        Returns
        -------
        ArrayLike
            An array-like with shape (n_samples, ) describing the respective
            labels predicted for each samples.

        """
        self.fit(X, y, X_val, y_val, **estimator_params)
        return self.predict(X)

    def save(self, output_path: PathLike) -> None:
        """Save the estimator to a file.

        Parameters
        ----------
        output_path : PathLike
            The path to save the estimator.

        """
        raise NotImplementedError
    
    @classmethod
    def load(cls, input_path: PathLike) -> "Estimator":
        """Load the estimator from a file.

        Parameters
        ----------
        input_path : PathLike
            The path to load the estimator.

        """
        raise NotImplementedError