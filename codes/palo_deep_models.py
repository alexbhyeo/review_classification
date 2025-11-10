import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Embedding, LSTM, 
                                   Conv1D, GlobalMaxPooling1D, 
                                   Bidirectional, Dropout, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

file_location = "w_review_train.csv"
# Load and preprocess the data

# Basic text cleaning function
def clean_text(text):
    if isinstance(text, str):
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove quotes if they surround the entire text
        text = text.strip('"')
    return text

# Parse the CSV data
def parse_data(filename):

    print(f"{'='*100}")
    reviews = pd.read_csv(filename, encoding='utf-8', sep=';', header=None, names=['review_text', 'rating'])
    print("length : ", len(reviews))
    print("header : ", reviews.dtypes)
    print(reviews.describe())

    print(f"{'='*100}")

    # Check for missing values
    print("Missing values:")
    print(reviews.isnull().sum())


    # Apply text cleaning
    reviews['cleaned_review'] = reviews['review_text'].apply(clean_text)

    return reviews

parsed_data = parse_data(file_location)



# Deep Learning Preprocessing
def prepare_dl_data(texts, ratings, max_features=10000, max_len=200):
    # Tokenize Thai text
    tokenizer = Tokenizer(
        num_words=max_features,
        oov_token='<OOV>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Convert ratings to categorical (1-5 to 0-4)
    y = to_categorical(np.array(ratings) - 1, num_classes=5)
    
    return X, y, tokenizer

# Prepare DL data
X_dl, y_dl, tokenizer = prepare_dl_data(
    parsed_data['cleaned_review'].values, 
    parsed_data['rating'].values
)

# Split for deep learning
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl
)

X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
    X_train_dl, y_train_dl, test_size=0.2, random_state=42
)


def create_hybrid_cnn_lstm(vocab_size, embedding_dim=128, max_len=200):
    inputs = Input(shape=(max_len,))
    
    # Embedding layer
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True
    )(inputs)
    
    # CNN branch
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(embedding)
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
    pooled1 = GlobalMaxPooling1D()(conv1)
    
    conv2 = Conv1D(64, 5, activation='relu', padding='same')(embedding)
    conv2 = Conv1D(64, 5, activation='relu', padding='same')(conv2)
    pooled2 = GlobalMaxPooling1D()(conv2)
    
    # LSTM branch
    lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    lstm = Bidirectional(LSTM(32))(lstm)
    
    # Concatenate all branches
    concatenated = Concatenate()([pooled1, pooled2, lstm])
    
    # Dense layers
    dense1 = Dense(128, activation='relu')(concatenated)
    dropout1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output layer
    outputs = Dense(5, activation='softmax')(dropout2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and display model
vocab_size = len(tokenizer.word_index) + 1
hybrid_model = create_hybrid_cnn_lstm(vocab_size)
hybrid_model.summary()

def create_multi_scale_cnn(vocab_size, embedding_dim=128, max_len=200):
    inputs = Input(shape=(max_len,))
    
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    
    # Multiple convolution branches with different kernel sizes
    conv_blocks = []
    for kernel_size in [2, 3, 4, 5]:
        conv = Conv1D(64, kernel_size, activation='relu', padding='same')(embedding)
        conv = Conv1D(64, kernel_size, activation='relu', padding='same')(conv)
        pooled = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pooled)
    
    # Concatenate all branches
    concatenated = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
    # Classification head
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

multi_cnn_model = create_multi_scale_cnn(vocab_size)

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
    min_lr=1e-7,
    verbose=1
)

# Train Hybrid CNN-LSTM
print("Training Hybrid CNN-LSTM Model...")
history_hybrid = hybrid_model.fit(
    X_train_dl, y_train_dl,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_dl, y_val_dl),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train Transformer Model
"""
print("Training Transformer Model...")
history_transformer = transformer_model.fit(
    X_train_dl, y_train_dl,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_dl, y_val_dl),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
"""

# Train Multi-Scale CNN
print("Training Multi-Scale CNN Model...")
history_multi_cnn = multi_cnn_model.fit(
    X_train_dl, y_train_dl,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_dl, y_val_dl),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


class DeepLearningEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return np.argmax(avg_prediction, axis=1) + 1  # Convert back to 1-5 scale
    
    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)

# Create ensemble
ensemble_models = [hybrid_model, multi_cnn_model]
deep_ensemble = DeepLearningEnsemble(ensemble_models)

from sklearn.metrics import f1_score, classification_report

def evaluate_deep_learning_models(models_dict, X_test, y_test):
    results = {}
    
    for name, model in models_dict.items():
        if name == 'Deep Ensemble':
            y_pred = model.predict(X_test)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1) + 1
        
        # Convert y_test back to 1-5 scale
        y_true = np.argmax(y_test, axis=1) + 1
        
        # Calculate metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        results[name] = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        print(f"  Per-class F1: {dict(zip(range(1,6), f1_per_class))}")
        print("-" * 50)
    
    return results

# Evaluate all deep learning models
dl_models = {
    'Hybrid CNN-LSTM': history_hybrid,
    'Multi-Scale CNN': history_multi_cnn,
    'Deep Ensemble': deep_ensemble
}

dl_results = evaluate_deep_learning_models(dl_models, X_test_dl, y_test_dl)