from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion(y_true, y_pred, label_path='grasp_labels_stable.csv'):
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
    print(f'Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}')

    cm = confusion_matrix(y_true, y_pred)
    labels = pd.read_csv(label_path)

    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    plt.imshow(cm, cmap='Blues')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    

    plt.xticks(np.arange(len(labels)), grasp_names,rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)), grasp_names)
    
    plt.show()