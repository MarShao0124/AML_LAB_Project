from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion(y_true, y_pred, label_path='grasp_labels_stable.csv',normalize=True):
    """
    绘制混淆矩阵
    
    参数:
        y_true (array): 真实标签
        y_pred (array): 预测标签
        label_path (str): 标签CSV文件路径

    返回:
        None
    """

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    cm = confusion_matrix(y_true, y_pred)

    # 归一化混淆矩阵
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 4)

    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    plt.imshow(cm, cmap='Blues')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')    

    plt.xticks(np.arange(len(labels)), grasp_names,rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)), grasp_names)
    
    plt.show()


def generate_classification_report(y_true, y_pred, label_path='grasp_labels_stable.csv'):

    # 读取类别标签
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    # 计算分类报告
    report_dict = classification_report(y_true, y_pred, target_names=grasp_names, output_dict=True)
    
    # 转换为 Pandas DataFrame
    df_report = pd.DataFrame(report_dict).T

    # **绘制热图**
    plt.figure(figsize=(10, 6))

    # **删除 "support" 列，避免影响可视化**
    df_report = df_report.drop(columns=['support'], errors='ignore')

    sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap="Blues", fmt=".4f")

    plt.title("Classification Report")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_history(history, acc='accuracy'):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    if acc == 'accuracy':
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
    else:
        plt.plot(history.history[acc], label='accuracy')
        plt.plot(history.history["val_"+acc], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')

    plt.show()