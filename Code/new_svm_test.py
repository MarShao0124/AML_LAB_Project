import pandas as pd
import numpy as np
import pickle
import serial
import time

# Function to load the saved scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Function to load the saved model
def load_model(model_path):
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise ValueError("请更新代码以支持其他类型的模型文件。")

# Real-time prediction function
def realtime_predict(model_path, scaler_path, arduino, label_path):
    # Load model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Load label mapping (assumes a CSV with columns 'Label' and 'Grasp Type')
    labels = pd.read_csv(label_path)
    labels_dict = dict(zip(labels['Label'], labels['Grasp Type']))
    
    print("模型已加载，正在监听 Arduino 数据...")

    try:
        while True:
            arduino.reset_input_buffer()  # 清空缓冲区，避免读取旧数据
            data = arduino.readline().decode('utf-8').strip()
            if not data:
                continue  # 跳过空数据

            # 输出原始数据，便于调试
            print("原始 Arduino 数据: {}".format(data))
            
            try:
                # 解析数据：将字符串转换为浮点数列表，并跳过空字符串
                data_list = [float(i) for i in data.split(',') if i.strip()]
                # 检查数据是否至少包含 7 个值（第一列为时间戳，其余为6维数据）
                if len(data_list) < 7:
                    print("警告: 数据长度不正确 ({}), 跳过该样本".format(len(data_list)))
                    continue  
                # 丢弃第一列（时间戳），只取后6维数据
                data_list = data_list[1:]
                if len(data_list) != 6:
                    print("警告: 数据长度不正确 ({}), 跳过该样本".format(len(data_list)))
                    continue
                # 可选：如果后6维中有低于50的值，则跳过该样本
                if any(x < 30 for x in data_list):
                    print("警告: 数据中存在低于30的值, 跳过该样本")
                    continue
                print("解析后的数据: {}".format(data_list))
            except ValueError:
                print("警告: 接收到格式错误的数据: {}".format(data))
                continue

            # 将数据转换为 NumPy 数组，并重塑为 (1,6)
            data_array = np.array(data_list).reshape(1, -1)
            
            # **Normalize the incoming data using the same scaler as in training**
           # data_array = scaler.transform(data_array)

            # 进行预测，并获取各类别概率（如果模型支持 predict_proba）
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_array)[0]
                predicted_label = np.argmax(probabilities)
            else:
                probabilities = None
                predicted_label = model.predict(data_array)[0]

            grasp_type = labels_dict.get(predicted_label, "Unknown Grasp Type")
            print("\npredict type: {} (label {})".format(grasp_type, predicted_label))
            
            if probabilities is not None:
                print("distribution:")
                for label, prob in zip(labels_dict.keys(), probabilities):
                    grasp = labels_dict.get(label, "Unknown Grasp Type")
                    print("  {} (label {}): {:.4f}".format(grasp, label, prob))
            
            time.sleep(0.5)  # 小延迟，避免过快输出

    except KeyboardInterrupt:
        print("\nstop ")
        arduino.close()

# Example usage:
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
model_path = 'Model/svm_model.pkl'
scaler_path = 'Model/scaler.pkl'
label_path = 'grasp_labels_stable.csv'
realtime_predict(model_path, scaler_path, arduino, label_path)
