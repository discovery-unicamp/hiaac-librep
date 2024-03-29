from librep.config.type_definitions import ArrayLike
from librep.base.parametrizable import Parametrizable


# Wrap around scikit learn base API
# Borrowed from Sklearn API


class Transform(Parametrizable):
    """For filtering or modifying the data, in a supervised or unsupervised way.
    `fit` allows implementing parametrizable transforms. This method sees the
    whole dataset. `transform` allows transforming each sample.
    """

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
        **fit_params,
    ) -> "Transform":
        """Fit the transformation with information of the whole dataset.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole dataset with shape:
            (n_samples, n_features).
        y : ArrayLike
            The respective labels, with shape: (n_samples, ). This parameter is
            optional and may be used if needed.
        X_val : ArrayLike
            An array-like representing the whole validation dataset with shape:
            (k_samples, n_features). This parameter is optional and may be used
            if needed.
        y_val : ArrayLike
            The respective validation labels, with shape: (k_samples, ). This
            parameter is optional and may be used if needed.
        **fit_params : type
            Optional data-dependent parameters.

        Returns
        -------
        'Transform'
            The transform (self).

        """
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transforms the dataset.

        Parameters
        ----------
        X : ArrayLike
            An array-like of sample with shape (n_samples, n_features, ).

        Returns
        -------
        ArrayLike
            An array-like with the transformed samples.

        """
        raise NotImplementedError

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike = None,
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
        **fit_params,
    ) -> ArrayLike:
        """Chain fit and transform methods, toghether. It firs the model and
        then transforms the training data.

        Parameters
        ----------
        X : ArrayLike
            An array-like representing the whole dataset with shape:
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

        Returns
        -------
        ArrayLike
            An array-like with the transformed samples.
        """
        self.fit(X, y, X_val, y_val, **fit_params)
        return self.transform(X)


class InvertibleTransform(Transform):
    """Denotes a invertible transform."""

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Perform the inverse transform on data.

        Parameters
        ----------
        X : ArrayLike
            An array-like of sample with shape (n_samples, n_features, ).

        Returns
        -------
        ArrayLike
            An array-like with the transformed samples.

        """
        raise NotImplementedError
