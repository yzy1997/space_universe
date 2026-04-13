import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif

DEFAULT_FEATURES = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
                    'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
                    'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE', 'ABSNJZH']


class MyPreprocessor:
    """
    Preprocessor and classifier for CME prediction.
    Uses RobustScaler for normalization, SelectKBest for feature selection,
    and SVM for classification.
    """
    def __init__(self, feature_columns=None, use_svm=True, k_best=12):
        """
        Initialize the preprocessor.

        Args:
            feature_columns: List of feature column names. If None, uses DEFAULT_FEATURES.
            use_svm: If True, use SVM classifier; if False, use LogisticRegression.
            k_best: Number of features to select with SelectKBest.
        """
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(DEFAULT_FEATURES)
        self.pipeline = None
        self.use_svm = use_svm
        self.k_best = k_best

    def transform(self, df):
        """Transform input data (DataFrame or array) to numeric array"""
        if isinstance(df, pd.DataFrame):
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise KeyError("Missing required feature columns: " + ", ".join(missing))

            x = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
            keep = x.dropna().index
            self.keep_index_ = keep
            return x.loc[keep]

        x = np.asarray(df)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != len(self.feature_columns):
            raise ValueError(f"Expected {len(self.feature_columns)} features, got {x.shape[1]}.")

        self.keep_index_ = np.arange(x.shape[0])
        return x

    def build_model(self):
        """Build the sklearn pipeline with scaler, feature selection, and classifier"""
        if self.use_svm:
            # Use SVM with optimized parameters for imbalanced data
            self.pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectKBest(score_func=f_classif, k=self.k_best)),
                ('svm', svm.SVC(
                    C=4.0,
                    gamma=0.075,
                    kernel='rbf',
                    class_weight={1: 6.5},  # Weight positive class more
                    cache_size=500,
                    max_iter=-1,
                    shrinking=True,
                    tol=1e-8,
                    probability=True
                ))
            ])
        else:
            # Use LogisticRegression as fallback
            self.pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('select', SelectKBest(score_func=f_classif, k=self.k_best)),
                ('lr', LogisticRegression(
                    C=1.0,
                    class_weight='balanced',
                    max_iter=5000
                ))
            ])
        return self.pipeline

    def fit(self, X, y):
        """Train the model"""
        if self.pipeline is None:
            self.build_model()
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.pipeline.predict_proba(X)

    def save(self, filepath):
        """Save the model"""
        if self.pipeline is None:
            raise ValueError("No model to save. Call build_model() or fit() first.")
        import joblib
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath):
        """Load a pre-trained model"""
        import joblib
        self.pipeline = joblib.load(filepath)
        return self.pipeline


def create_bundle(clf, preprocessor, target="CME"):
    """Create the submission bundle dictionary"""
    return {
        "model": clf,
        "preprocessor": preprocessor,
        "target": target
    }