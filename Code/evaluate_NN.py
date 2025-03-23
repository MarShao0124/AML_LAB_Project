import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from load_data import load_data  # 确保这个函数可以正确加载数据

# 配置文件路径
model_path = 'Model/NN_model.h5'
DATA_PATH = 'Stable_dataset.csv'
LABEL_PATH = 'grasp_labels_stable.csv'


def plot_confusion_matrix(y_true, y_pred, label_path=LABEL_PATH, normalize=True, save_path="confusion_matrix.png"):

    cm = confusion_matrix(y_true, y_pred)
    
    # 读取类别标签
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    # 计算准确率 & F1 分数
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    # 归一化混淆矩阵
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=grasp_names, yticklabels=grasp_names)

    # **调整 x 轴标签的角度，防止重叠**
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # **优化布局，防止标签被裁剪**
    plt.tight_layout()

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    # 保存图片
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix saved to {save_path}")
    plt.show()


def generate_classification_report(y_true, y_pred, label_path=LABEL_PATH, save_csv_path="classification_report.csv", save_img_path="classification_report.png"):

    # 读取类别标签
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    # 计算分类报告
    report_dict = classification_report(y_true, y_pred, target_names=grasp_names, output_dict=True)
    
    # 转换为 Pandas DataFrame
    df_report = pd.DataFrame(report_dict).T

    # 保存 CSV 文件
    df_report.to_csv(save_csv_path, index=True)
    print(f"Classification report saved to {save_csv_path}")

    # **绘制热图**
    plt.figure(figsize=(10, 6))

    # **删除 "support" 列，避免影响可视化**
    df_report = df_report.drop(columns=['support'], errors='ignore')

    sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap="Blues", fmt=".2f")

    plt.title("Classification Report")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_img_path, dpi=300)
    print(f"Classification report image saved to {save_img_path}")
    plt.show()


def main():
    # ** 加载 NN.h5 训练好的模型 **
    # 加载 NN.h5 训练好的模型
    classifier = load_model(model_path)
    print(f"Model loaded from {model_path}")
   

    # ** 加载测试数据 **
    (X_train, y_train), (X_test, y_test) = load_data(DATA_PATH, label_columns='Labels', test_size=0.2)

    # 丢弃时间戳列（如果数据包含时间戳）
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    # ** 进行预测 **
    y_pred = classifier.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # ** 绘制 Confusion Matrix **
    plot_confusion_matrix(y_test, y_pred, label_path=LABEL_PATH, normalize=True, save_path="confusion_matrix.png")

    # ** 生成分类报告热图 **
    generate_classification_report(y_test, y_pred, label_path=LABEL_PATH, save_csv_path="classification_report.csv", save_img_path="classification_report.png")


if __name__ == "__main__":
    main()
