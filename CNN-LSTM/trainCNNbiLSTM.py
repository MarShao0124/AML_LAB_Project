#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trainCNNLSTM.py
Training a CNN + BiLSTM model for 6D sensor data classification with 8 grasp types.
Now modified to save the trained model as a .pkl file with fixed random seeds.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths and hyperparameters
OUTPUT_DIR = "./CNNbiLSTM"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_lstm_model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
STEP_SIZE = 5
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 42  # Fixed seed for reproducibility

# ✅ **Set random seeds for reproducibility**
def set_random_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ✅ **Set the seeds before anything else**
set_random_seeds()

def sliding_window_subsequences(data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Apply a sliding window to generate subsequences from time-series data."""
    subseqs = [data[start:start + window_size] for start in range(0, len(data) - window_size + 1, step_size)]
    return subseqs

def load_dataset(data_dir="./ProcessedData"):
    """Load dataset, apply sliding window, and split into train/val/test sets."""
    grasp_folders = [
        "Adduction_Grip", "Fixed_hook", "Large_diameter", "Parallel_extension",
        "Precision_sphere", "Sphere_4_finger", "Ventral", "Writing_Tripod"
    ]
    X_list, y_list = [], []

    for label_name in grasp_folders:
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder_path):
            continue

        for subf in os.listdir(folder_path):
            subf_path = os.path.join(folder_path, subf)
            csv_files = glob.glob(os.path.join(subf_path, '*.csv'))

            for file_path in csv_files:
                df = pd.read_csv(file_path, header=None)
                df.columns = ['time', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
                data = df.iloc[:, 1:].values  # Extract sensor readings

                subseqs = sliding_window_subsequences(data, WINDOW_SIZE, STEP_SIZE)
                X_list.extend(subseqs)
                y_list.extend([label_name] * len(subseqs))

    X, y = np.array(X_list), np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_onehot = to_categorical(y_enc, num_classes=NUM_CLASSES)

    # Split dataset: 80% train, 10% val, 10% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=SEED, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1111, random_state=SEED, stratify=y_trainval)

    print("Dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, le

def build_cnn_lstm_model(input_shape=(WINDOW_SIZE, 6), num_classes=NUM_CLASSES):
    """Builds a CNN + BiLSTM model."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def save_model_as_pkl(model, filename):
    """Save the trained model as a .pkl file."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def plot_classification_report(report, labels, save_path):
    """Generate a classification report plot and save it."""
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :]
    report_df['Accuracy (%)'] = (report_df['recall'] * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.6 + 2))
    ax.axis('off')
    table = ax.table(cellText=report_df.round(2).values,
                     colLabels=['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy (%)'],
                     rowLabels=labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Classification Report", fontsize=14, weight='bold')

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved classification report to {save_path}")
    plt.close()

def plot_confusion_matrix(cm, labels, save_path):
    """Generate a confusion matrix plot and save it."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

def main():
    """Main function to train, evaluate, and save the CNN + BiLSTM model."""
    set_random_seeds()  # Ensure reproducibility

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_dataset()
    model = build_cnn_lstm_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy = {test_acc:.4f}")

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Save classification report and confusion matrix
    plot_classification_report(report, label_encoder.classes_, os.path.join(OUTPUT_DIR, "classification_report.png"))
    plot_confusion_matrix(cm, label_encoder.classes_, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Save model as .pkl file
    save_model_as_pkl(model, MODEL_PATH)

if __name__ == "__main__":
    main()
