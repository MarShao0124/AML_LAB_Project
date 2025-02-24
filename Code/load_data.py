import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def load_data(
    csv_path,
    label_columns=None,
    test_size=0.2,
    task='classification',
    seed=42,
    normalize=True,
    shuffle=True
):
    """
    加载并预处理数据集，返回适合Keras/TensorFlow的训练测试分割
    
    参数:
        csv_path (str): CSV文件路径
        label_columns (str/list): 标签列名，默认最后一列
        test_size (float): 测试集比例 (0.0-1.0)
        task (str): 任务类型 'classification' 或 'regression'
        seed (int): 随机种子
        normalize (bool): 是否标准化特征数据
        shuffle (bool): 是否打乱数据
    
    返回:
        tuple: ((X_train, y_train), (X_test, y_test))
    
    异常:
        ValueError: 无效的任务类型或标签列错误
    """
    try:
        # 读取数据
        df = pd.read_csv(csv_path)
        
        # 分离特征和标签
        if label_columns is None:
            y = df.iloc[:, -1].values  # 默认最后一列为标签
            X = df.iloc[:, :-1].values
        else:
            y = df[label_columns].values
            X = df.drop(label_columns, axis=1).values
            
        # 数据标准化
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
        # 处理分类标签
        if task == 'classification':
            # 编码字符串标签为整数
            if y.dtype == np.object_:
                le = LabelEncoder()
                y = le.fit_transform(y)
            

        elif task != 'regression':
            raise ValueError(f"无效任务类型: {task}，请选择 'classification' 或 'regression'")
            
        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle
        )
        
        # 转换为float32类型
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        return (X_train, y_train), (X_test, y_test)
    
    except Exception as e:
        raise RuntimeError(f"数据加载失败: {str(e)}") from e