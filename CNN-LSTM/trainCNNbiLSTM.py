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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths and hyperparameters
OUTPUT_DIR = "./CNNbiLSTM"
MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_lstm_model.pkl")
LABEL_PATH = "grasp_labels_stable.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
STEP_SIZE = 5
NUM_CLASSES = 8
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 42  

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
    return [data[start:start + window_size] for start in range(0, len(data) - window_size + 1, step_size)]

def load_dataset(data_dir="ProcessedData_overall"):
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
                data = df.iloc[:, 1:].values 

                subseqs = sliding_window_subsequences(data, WINDOW_SIZE, STEP_SIZE)
                X_list.extend(subseqs)
                y_list.extend([label_name] * len(subseqs))

    X, y = np.array(X_list), np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_onehot = to_categorical(y_enc, num_classes=NUM_CLASSES)

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

def plot_confusion_matrix(y_true, y_pred, label_path=LABEL_PATH, normalize=True, save_path="confusion_matrix.png"):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=grasp_names, yticklabels=grasp_names)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix saved to {save_path}")
    plt.show()

def generate_classification_report(y_true, y_pred, save_csv_path="classification_report.csv", save_img_path="classification_report.png"):
    """Generate and save the classification report as CSV and image."""
    labels = pd.read_csv(LABEL_PATH)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    report_dict = classification_report(y_true, y_pred, target_names=grasp_names, output_dict=True)
    
    df_report = pd.DataFrame(report_dict).T
    df_report.to_csv(save_csv_path, index=True)
    print(f"Classification report saved to {save_csv_path}")

    plt.figure(figsize=(10, 6))
    df_report = df_report.drop(columns=['support'], errors='ignore')

    sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_img_path, dpi=300)
    print(f"Classification report image saved to {save_img_path}")
    plt.show()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_dataset()
    model = build_cnn_lstm_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[early_stop])

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    plot_confusion_matrix(y_true, y_pred)
    generate_classification_report(y_true, y_pred)

    save_model_as_pkl(model, MODEL_PATH)

if __name__ == "__main__":
    main()
