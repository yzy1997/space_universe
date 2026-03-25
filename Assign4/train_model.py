#!/usr/bin/env python3
"""
Improved CNN for Exoplanet Detection
Uses GPU:1 and has a much better architecture for higher accuracy
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')
np.random.seed(42)

# USE GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus}")
else:
    print("No GPU found, using CPU")

# Load and prepare dataset
full_train_data = pd.read_csv('/home/yangz2/code/space_universe/Assign4/exoTrain.csv')
full_train_data['LABEL'] = full_train_data['LABEL'].map({1: 0, 2: 1})

X = full_train_data.drop(columns=['LABEL'])
y = full_train_data['LABEL'].astype(int)

# Student-visible split: train / validation / public test
X_train_raw, X_other_raw, y_train_raw, y_other_raw = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val_raw, X_public_raw, y_val, y_public = train_test_split(
    X_other_raw, y_other_raw, test_size=0.50, random_state=42, stratify=y_other_raw
)

print(f'Student train raw:   {len(X_train_raw)}')
print(f'Student val raw:     {len(X_val_raw)}')
print(f'Student public test: {len(X_public_raw)}')

# Enhanced preprocessing with optimized parameters
# Try different sigma values to find optimal
sigma = 5  # Reduced sigma for better feature preservation

X_train = normalize(X_train_raw.to_numpy())
X_val = normalize(X_val_raw.to_numpy())
X_public = normalize(X_public_raw.to_numpy())

X_train = ndimage.gaussian_filter(X_train, sigma=sigma)
X_val = ndimage.gaussian_filter(X_val, sigma=sigma)
X_public = ndimage.gaussian_filter(X_public, sigma=sigma)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_val = std_scaler.transform(X_val)
X_public = std_scaler.transform(X_public)

train_X, train_y = X_train, y_train_raw.to_numpy()
val_X, val_y = X_val, y_val.to_numpy()

print(f'train_X: {train_X.shape}, train_y: {train_y.shape}')
print(f'val_X:   {val_X.shape}, val_y:   {val_y.shape}')
print(f'Class distribution in train: {np.bincount(train_y)}')

# Reshape for CNN
train_X_reshaped = np.asarray(train_X, dtype=np.float32).reshape(-1, train_X.shape[1], 1)
val_X_reshaped = np.asarray(val_X, dtype=np.float32).reshape(-1, val_X.shape[1], 1)
public_X_reshaped = np.asarray(X_public, dtype=np.float32).reshape(-1, X_public.shape[1], 1)

# Build improved CNN model
# Deeper network with more filters, BatchNorm, and Dropout
tf.keras.backend.clear_session()

time_steps = train_X.shape[1]

# Input
inputs = layers.Input(shape=(time_steps, 1))

# First conv block
x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.3)(x)

# Second conv block
x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.3)(x)

# Third conv block
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

# Fourth conv block
x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.5)(x)

# Dense layers
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Output
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile with Adam optimizer and lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train with all data and more epochs
# Use class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Use all training samples
print("Training with all data...")
history = model.fit(
    train_X_reshaped,
    train_y,
    epochs=30,
    batch_size=32,
    validation_data=(val_X_reshaped, val_y),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save preprocessor class
preprocessor_code = '''
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
            # Keras API differs across versions; support both paths.
            try:
                self._model = tf.keras.models.model_from_config(self.config)
            except AttributeError:
                self._model = tf.keras.Sequential.from_config(self.config)
            self._model.set_weights(self.weights)
            self._model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def predict(self, X):
        self._ensure_model()
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        arr = arr.reshape(-1, arr.shape[1], 1)
        prob = self._model.predict(arr, verbose=0).ravel()
        return (prob >= self.threshold).astype(int)
'''

# Write the preprocessor module
with open('/home/yangz2/code/space_universe/Assign4/preprocessor_class.py', 'w', encoding='utf-8') as f:
    f.write(preprocessor_code)

print("Saved preprocessor_class.py")

# Import from the module
import sys
sys.path.insert(0, '/home/yangz2/code/space_universe/Assign4')
from preprocessor_class import MyPreprocessor, CNNJoblibModel

TARGET = 'LABEL'
FEATURES = list(X_train_raw.columns)

# Use sigma=5 to match training
bundle = {
    'target': TARGET,
    'model': CNNJoblibModel(model.get_config(), model.get_weights(), threshold=0.5),
    'preprocessor': MyPreprocessor(FEATURES, std_scaler.mean_, std_scaler.scale_, sigma=5),
}

joblib.dump(bundle, '/home/yangz2/code/space_universe/Assign4/student_model.joblib')
print('Saved student_model.joblib')

# Validate model
val_prob = model.predict(val_X_reshaped, verbose=0).ravel()
val_pred = (val_prob >= 0.5).astype(int)

public_prob = model.predict(public_X_reshaped, verbose=0).ravel()
public_pred = (public_prob >= 0.5).astype(int)

print('\n=== Validation Results ===')
print(f'Validation Accuracy: {accuracy_score(val_y, val_pred):.4f}')
print(f'Validation F1-score: {f1_score(val_y, val_pred):.4f}')
print(classification_report(val_y, val_pred))

print('\n=== Public Test Results ===')
print(f'Public Accuracy: {accuracy_score(y_public, public_pred):.4f}')
print(f'Public F1-score: {f1_score(y_public, public_pred):.4f}')
print(classification_report(y_public, public_pred))

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training vs Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('/home/yangz2/code/space_universe/Assign4/accuracy_plot.png', dpi=150)
plt.close()

# Confusion matrix (validation)
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(val_y, val_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/home/yangz2/code/space_universe/Assign4/confusion_matrix.png', dpi=150)
plt.close()

# ROC curve (validation)
fpr, tpr, _ = roc_curve(val_y, val_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('/home/yangz2/code/space_universe/Assign4/roc_curve.png', dpi=150)
plt.close()

print("\nPlots saved!")

# Test bundle loading
print("\n=== Testing bundle loading ===")
bundle_loaded = joblib.load('/home/yangz2/code/space_universe/Assign4/student_model.joblib')
test_model = bundle_loaded['model']
test_preprocessor = bundle_loaded['preprocessor']

X_eval_df = X_public_raw.copy()
X_eval_processed = test_preprocessor.transform(X_eval_df)
y_eval_series = pd.Series(y_public).astype(int)
y_eval_aligned = y_eval_series.loc[X_eval_processed.index]

y_pred = test_model.predict(X_eval_processed)
acc = accuracy_score(y_eval_aligned, y_pred)
f1 = f1_score(y_eval_aligned, y_pred)
print(f'Bundle test accuracy: {acc:.4f}')
print(f'Bundle test F1: {f1:.4f}')

print("\n=== Done! ===")