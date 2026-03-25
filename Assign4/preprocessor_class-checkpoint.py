
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.preprocessing import normalize

class MyPreprocessor:
    def __init__(self, feature_columns, scaler_mean, scaler_scale, sigma=10):
        self.feature_columns = list(feature_columns)
        self.scaler_mean_ = np.asarray(scaler_mean)
        self.scaler_scale_ = np.asarray(scaler_scale)
        self.sigma = sigma

    def transform(self, df):
        df = df.copy()
        X = df[self.feature_columns]

        # Drop rows with missing values and preserve indices
        keep = X.dropna().index
        self.keep_index_ = keep
        X = X.loc[keep]

        Xn = normalize(X.to_numpy())
        Xg = ndimage.gaussian_filter(Xn, sigma=self.sigma)
        Xs = (Xg - self.scaler_mean_) / self.scaler_scale_
        return pd.DataFrame(Xs, index=keep, columns=self.feature_columns)

class CNNJoblibModel:
    def __init__(self, config, weights, threshold=0.5):
        self.config = config
        self.weights = weights
        self.threshold = threshold
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import os as _os2
            _os2.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            _os2.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            _os2.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            # Keras API differs across versions; support both paths.
            try:
                self._model = tf.keras.models.model_from_config(self.config)
            except AttributeError:
                self._model = tf.keras.Sequential.from_config(self.config)
            self._model.set_weights(self.weights)
            self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def predict(self, X):
        self._ensure_model()
        arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        arr = arr.reshape(-1, arr.shape[1], 1)
        prob = self._model.predict(arr, verbose=0).ravel()
        return (prob >= self.threshold).astype(int)
