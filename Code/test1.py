#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test1.py
研究生机器学习 Lab 项目示例：
利用 6 个 flex 传感器数据对 8 种抓握姿势进行分类与聚类分析。
包含：
1. 数据读取（递归读取子文件夹）
2. 标准化
3. 可视化（时序曲线 + PCA）
4. 聚类 (K-Means)
5. 分类 (SVM/RandomForest)
6. 图片保存功能
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 创建保存图像的文件夹
PLOT_DIR = ".\Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data(data_dir, grasp_folders):
    """
    从指定文件夹中读取 CSV 数据，并返回拼接后的 DataFrame。
    每个 grasp 文件夹下有若干子文件夹（对应不同物品），再在子文件夹中存放 CSV 文件。
    假设 CSV 格式：
        time, flex1, flex2, flex3, flex4, flex5, flex6
    在返回的 DataFrame 中新增一列 'label' 作为抓握类别标识。

    :param data_dir: 数据根目录 (例如 "./ProcessedData")
    :param grasp_folders: 一个列表，每个元素是一个抓握类别的文件夹名称
    :return: pandas DataFrame, 包含所有数据以及抓握标签
    """
    all_data = []
    for grasp_name in grasp_folders:
        folder_path = os.path.join(data_dir, grasp_name)
        print(f"Searching in: {folder_path}")

        # 列出该抓握姿势文件夹下的所有子文件夹（对应不同物体）
        # 例如 chopstick, pen, stick 等
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory.")
            continue

        subfolders = [sf for sf in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, sf))]

        # 对于每个子文件夹，获取其中所有 CSV
        for subf in subfolders:
            subf_path = os.path.join(folder_path, subf)
            # 匹配所有 csv 文件
            csv_files = glob.glob(os.path.join(subf_path, '*.csv'))

            # 你也可以改成递归匹配:
            # csv_files = glob.glob(os.path.join(subf_path, '**', '*.csv'), recursive=True)

            print(f"  - Subfolder '{subf}' has {len(csv_files)} csv files.")

            for file_path in csv_files:
                # 读取 CSV
                df = pd.read_csv(file_path, header=None)
                # 根据 CSV 实际格式设置列名
                df.columns = ['time', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']

                # 如果不需要 time，可以注释掉下一行
                # df = df.drop(columns=['time'])

                # 为该 df 添加 label，表示抓握姿势
                # 如果你想把不同物品分开，可以用 f"{grasp_name}_{subf}"
                # 但这里我们只区分抓握姿势，不区分物品
                df['label'] = grasp_name

                all_data.append(df)

    if not all_data:
        raise ValueError("No data loaded. Check file paths, extensions, or CSV content.")

    # 拼接所有数据
    all_data_df = pd.concat(all_data, ignore_index=True)
    return all_data_df


def visualize_time_series(df, grasp_folders, sample_size=100):
    """
    对部分时序数据做可视化。
    这里演示：从每个类别随机取一段数据进行时序曲线展示。
    :param df: 完整的 DataFrame
    :param grasp_folders: 类别名称列表
    :param sample_size: 取多少行数据来展示
    """
    fig, axs = plt.subplots(len(grasp_folders), 1, figsize=(10, 12), sharex=True)
    if len(grasp_folders) == 1:
        axs = [axs]  # 保证可迭代

    for i, grasp_name in enumerate(grasp_folders):
        # 从 df 中筛选该类别数据
        subset = df[df['label'] == grasp_name]
        # 随机取 sample_size 行
        subset_sample = subset.sample(min(sample_size, len(subset)), random_state=42)
        # 按照 time 排序
        subset_sample = subset_sample.sort_values(by='time')

        for col in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']:
            axs[i].plot(subset_sample['time'], subset_sample[col], label=col)

        axs[i].set_title(f"Time Series of {grasp_name}")
        axs[i].legend(loc='best')

    save_path = os.path.join(PLOT_DIR, f"time_series_{grasp_name}.png")
    plt.tight_layout()    
    plt.savefig(save_path)
    print(f"Saved time series plot to {save_path}")
    plt.show()


def standardize_data(df):
    """
    对传感器数据进行标准化处理（z-score）。
    :param df: 包含传感器列的 DataFrame
    :return: (scaled_features, scaler)
    """
    feature_cols = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
    scaler = StandardScaler()

    # 拿到特征矩阵
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def visualize_pca(X_scaled, labels):
    """
    利用 PCA 将 6 维数据降到 2 维，并可视化不同抓握姿势的分布。
    :param X_scaled: 标准化后的特征矩阵
    :param labels: 抓握类别标签
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", s=60)
    plt.title("PCA Visualization of Grasp Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "pca_visualization.png")
    plt.savefig(save_path)
    print(f"Saved PCA plot to {save_path}")
    plt.show()


def cluster_data(X_scaled, n_clusters=8):
    """
    对数据进行聚类 (K-Means)，并计算轮廓系数 (silhouette score)。
    :param X_scaled: 标准化后的特征矩阵
    :param n_clusters: 期望聚类数
    :return: cluster_labels, silhouette
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, cluster_labels)

    print(f"K-Means with {n_clusters} clusters, silhouette score = {sil_score:.3f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="Set1", s=60)
    plt.title(f"K-Means Clustering (n_clusters={n_clusters})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "kmeans_clusters.png")
    plt.savefig(save_path)
    print(f"Saved K-Means clustering plot to {save_path}")
    plt.show()

    return cluster_labels, sil_score


def classification_experiment(X_scaled, labels, method='SVM'):
    """
    使用给定的分类器对数据进行训练和评估。
    支持 SVM 或 RandomForest。
    :param X_scaled: 标准化后的特征矩阵
    :param labels: 抓握类别标签
    :param method: 'SVM' 或 'RF'
    """
    # 将标签转换成数值或保持字符串均可
    # 如果是字符串标签，可以直接传给 sklearn，会自动编码
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    if method == 'SVM':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    elif method == 'RF':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("method must be 'SVM' or 'RF'")

    # 训练
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 评估
    print(f"=== Classification Report ({method}) ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels),
                yticklabels=np.unique(labels))
    plt.title(f"Confusion Matrix ({method})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_path = os.path.join(PLOT_DIR, f"confusion_matrix_{method}.png")
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")
    plt.show()


def main():
    # 1. 设置数据路径和抓握类别
    # 假设 test1.py 与 ProcessedData 位于同级目录
    # 如果不在同级，请根据实际情况修改 data_dir 的相对或绝对路径
    data_dir = ".\ProcessedData"  # 根据你的实际情况改动

    # 8 类抓握姿势文件夹（需与实际文件夹名保持一致，大小写一致）
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

    # 2. 读取数据
    df = load_data(data_dir, grasp_folders)
    print("Data shape:", df.shape)
    print(df.head())

    # 3. 简单可视化：时序曲线
    visualize_time_series(df, grasp_folders, sample_size=50)

    # 4. 数据标准化
    X_scaled, scaler = standardize_data(df)
    labels = df['label'].values  # 类别标签

    # 5. 降维可视化（PCA）
    visualize_pca(X_scaled, labels)

    # 6. 聚类分析 (K-Means)
    cluster_labels, sil_score = cluster_data(X_scaled, n_clusters=8)

    # 7. 分类实验（SVM）
    classification_experiment(X_scaled, labels, method='SVM')

    # 8. 分类实验（Random Forest）
    classification_experiment(X_scaled, labels, method='RF')

    # 你也可以在此尝试更多模型或做更多对比


if __name__ == "__main__":
    main()
