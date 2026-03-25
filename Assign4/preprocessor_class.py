import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.preprocessing import normalize


class MyPreprocessor:
    def __init__(self, feature_columns, scaler_mean, scaler_scale, sigma=3.0, eps=1e-8):
        self.feature_columns = list(feature_columns)
        self.scaler_mean_ = np.asarray(scaler_mean, dtype=np.float32)
        self.scaler_scale_ = np.asarray(scaler_scale, dtype=np.float32)
        self.sigma = float(sigma)
        self.eps = float(eps)

    @classmethod
    def fit_from_training_df(cls, df, feature_columns, sigma=2.0, eps=1e-8):
        feature_columns = list(feature_columns)
        X = df[feature_columns].copy()
        X = X.dropna()

        arr = X.to_numpy(dtype=np.float32)
        arr = normalize(arr, norm='l2', axis=1)
        # 关键修改：只沿特征维(axis=1)平滑，不同事例之间
        arr = ndimage.gaussian_filter1d(arr, sigma=sigma, axis=1, mode='nearest')

        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale = np.where(scale < eps, 1.0, scale)

        return cls(feature_columns, mean, scale, sigma=sigma, eps=eps)

    def transform(self, df):
        df = df.copy()
        X = df[self.feature_columns].copy()

        keep = X.dropna().index
        X = X.loc[keep]

        arr = X.to_numpy(dtype=np.float32)
        arr = normalize(arr, norm='l2', axis=1)
        # 只沿特征维平滑
        arr = ndimage.gaussian_filter1d(arr, sigma=self.sigma, axis=1, mode='nearest')
        arr = (arr - self.scaler_mean_) / np.where(self.scaler_scale_ < self.eps, 1.0, self.scaler_scale_)

        return pd.DataFrame(arr, index=keep, columns=self.feature_columns)


class CNNJoblibModel:
    def __init__(self, config, weights, threshold=0.5):
        self.config = config
        self.weights = weights
        self.threshold = float(threshold)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import os as _os2
            _os2.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
            _os2.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            _os2.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

            import tensorflow as tf
            try:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    try:
                        tf.config.set_visible_devices(gpus[0], "GPU")
                    except RuntimeError:
                        pass
            except Exception:
                pass

            try:
                self._model = tf.keras.models.model_from_config(self.config)
            except AttributeError:
                self._model = tf.keras.Sequential.from_config(self.config)

            self._model.set_weights(self.weights)
            self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def predict(self, X):
        self._ensure_model()
        arr = X.to_numpy(dtype=np.float32) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float32)
        arr = arr.reshape(-1, arr.shape[1], 1)
        prob = self._model.predict(arr, verbose=0).ravel()
        return (prob >= self.threshold).astype(int)