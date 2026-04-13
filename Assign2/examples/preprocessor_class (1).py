import numpy as np
import pandas as pd

DEFAULT_FEATURES = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH', 'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE', 'ABSNJZH']


class MyPreprocessor:
    """
    Preprocessor used for the A2 submission bundle.

    - If input is a pandas DataFrame, select the required SHARP features,
      coerce them to numeric values, and drop rows with missing values.
    - If input is already a NumPy-like array, pass it through after a shape check.
    """

    def __init__(self, feature_columns=None):
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(DEFAULT_FEATURES)

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise KeyError(
                    "Missing required feature columns: " + ", ".join(missing)
                )

            x = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
            keep = x.dropna().index
            self.keep_index_ = keep
            return x.loc[keep]

        x = np.asarray(df)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {x.shape[1]}."
            )

        self.keep_index_ = np.arange(x.shape[0])
        return x
