import tensorflow as tf
from tensorflow import keras


class MyPreprocessor:
    """
    CNN+GRU model for flare detection in TESS lightcurves.
    This class builds, trains, and saves the neural network model.
    """
    def __init__(self, cadences=200):
        self.cadences = cadences
        self.model = None
        self.history = None

    def build_model(self):
        """Build the CNN+GRU hybrid model"""
        keras.backend.clear_session()
        tf.random.set_seed(42)

        self.model = keras.Sequential([
            keras.layers.Input(shape=(self.cadences, 1)),

            # Local pattern extraction
            keras.layers.Conv1D(
                filters=16,
                kernel_size=7,
                padding='same',
                activation='relu'
            ),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.1),

            keras.layers.Conv1D(
                filters=32,
                kernel_size=5,
                padding='same',
                activation='relu'
            ),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.1),

            # Temporal aggregation
            keras.layers.GRU(
                32,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.0
            ),

            # Classifier
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='Adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def fit(self, train_data, train_labels, val_data=None, val_labels=None,
            epochs=200, batch_size=32, verbose=1):
        """Train the model"""
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
        ]

        validation_data = (val_data, val_labels) if val_data is not None else None

        self.history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict(self, data):
        """Predict flare probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.model.predict(data)

    def save(self, filepath):
        """Save the model"""
        if self.model is not None:
            self.model.save(filepath)

    def load(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        return self.model