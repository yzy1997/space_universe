from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression


class MyPreprocessor:
    """
    Preprocessor and classifier for CME prediction.
    Uses RobustScaler for normalization and LogisticRegression for classification.
    """
    def __init__(self):
        self.pipeline = None

    def build_model(self):
        """Build the sklearn pipeline with scaler and classifier"""
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
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