import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from load_data import load_data  # 确保这个函数可以正确加载数据

# 配置文件路径
MODEL_PATH = 'Model/new_svm_model.pkl'
DATA_PATH = 'Stable_dataset.csv'
LABEL_PATH = 'grasp_labels_stable.csv'

def plot_confusion_matrix(y_true, y_pred, label_path='grasp_labels_stable.csv', normalize=True, save_path="confusion_matrix.png"):
    """
    绘制混淆矩阵（支持归一化）
    
    参数:
        y_true (array): 真实标签
        y_pred (array): 预测标签
        label_path (str): 标签CSV文件路径
        normalize (bool): 是否进行归一化
        save_path (str): 生成的图片保存路径

    返回:
        None
    """
    # 计算混淆矩阵
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
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # 保存图片
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix saved to {save_path}")
    plt.show()


def main():
    # ** 加载 SVM 训练好的模型 **
    with open(MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    print("Loaded trained SVM model.")

    # ** 加载测试数据 **
    (X_train, y_train), (X_test, y_test) = load_data(DATA_PATH, label_columns='Labels', test_size=0.2)
    
    # 丢弃时间戳列（如果数据包含时间戳）
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    print("First 5 rows of X_train:")
    print(X_train[:5, :])
    print("First 5 labels of y_train:")
    print(y_train[:5])

    # ** 进行预测 **
    y_pred = classifier.predict(X_test)

    # ** 绘制 Confusion Matrix **
    plot_confusion_matrix(y_test, y_pred, label_path=LABEL_PATH, normalize=False, save_path="confusion_matrix.png")

if __name__ == "__main__":
    main()
