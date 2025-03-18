#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trainCNN.py
示例：使用 1D CNN + 滑动窗口 对 6 维传感器数据进行 8 类抓握姿势分类
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 路径设置
OUTPUT_DIR = "./CNN"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 超参数
WINDOW_SIZE = 10   # 滑动窗口大小
STEP_SIZE = 5      # 滑动步长
NUM_CLASSES = 8     # 8 种抓握姿势
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

def sliding_window_subsequences(data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    对单个时间序列 data (形状: (T,6)) 进行滑动窗口切片，生成多个子序列。
    每个子序列形状: (window_size, 6)。
    如果最后剩余长度 < window_size，则直接跳过。
    """
    subseqs = []
    T = len(data)
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        subseqs.append(data[start:end])
    return subseqs

def load_dataset(data_dir=".\ProcessedData"):
    """
    读取 CSV 文件，并使用滑动窗口生成多个样本。
    输出: X (N, window_size, 6), y (N,)
    """
    grasp_folders = [
        "Adduction_Grip",
        "Fixed_hook",
        "Large_diameter",
        "Parallel_extension",
        "Precision_sphere",
        "Sphere_4_finger",
        "Ventral",
        "Writing_Tripod"
    ]
    X_list = []
    y_list = []

    for label_name in grasp_folders:
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder_path):
            continue

        subfolders = [sf for sf in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, sf))]

        for subf in subfolders:
            subf_path = os.path.join(folder_path, subf)
            csv_files = glob.glob(os.path.join(subf_path, '*.csv'))

            for file_path in csv_files:
                df = pd.read_csv(file_path, header=None)
                df.columns = ['time','flex1','flex2','flex3','flex4','flex5','flex6']

                # 取出传感器数据 (T, 6)
                data = df[['flex1','flex2','flex3','flex4','flex5','flex6']].values

                # 滑动窗口生成子序列
                subseqs = sliding_window_subsequences(data, WINDOW_SIZE, STEP_SIZE)
                for subseq in subseqs:
                    X_list.append(subseq)
                    y_list.append(label_name)

    X = np.array(X_list)  # (N, window_size, 6)
    y = np.array(y_list)  # (N, )

    # 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_onehot = to_categorical(y_enc, num_classes=NUM_CLASSES)

    # 划分训练/验证/测试集 (8:1:1)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_onehot, test_size=0.1, random_state=42, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1111, random_state=42, stratify=y_trainval
    )

    print("Dataset shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, le

def build_cnn_model(input_shape=(WINDOW_SIZE, 6), num_classes=NUM_CLASSES):
    """
    简单 1D CNN 模型
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def plot_classification_report(report, labels, save_path):
    """
    绘制分类报告表格并保存为图片。
    """
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :]  # 去掉accuracy等行
    report_df['Accuracy (%)'] = (report_df['recall'] * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.6 + 2))
    ax.axis('off')
    table = ax.table(cellText=report_df.round(2).values,
                     colLabels=['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy (%)'],
                     rowLabels=labels,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Classification Report", fontsize=14, weight='bold')

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved classification report to {save_path}")
    plt.close()

def plot_confusion_matrix(cm, labels, save_path):
    """
    绘制混淆矩阵并保存为图片。
    """
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
    # 1. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_dataset()

    # 2. 构建模型
    model = build_cnn_model()

    # 3. 训练
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2
    )

    # 4. 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy = {test_acc:.4f}")

    # 5. 分类报告和混淆矩阵
    # 预测与报告
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # 保存分类报告和混淆矩阵
    plot_classification_report(report, label_encoder.classes_, os.path.join(OUTPUT_DIR, "classification_report.png"))
    plot_confusion_matrix(cm, label_encoder.classes_, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

if __name__ == "__main__":
    main()
