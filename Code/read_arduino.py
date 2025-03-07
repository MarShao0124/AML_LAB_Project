import serial 
import time
import pickle
import pandas as pd
import numpy as np

# 初始化串口连接
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
model_path = 'Model/svm_model.pkl'
label_path = 'grasp_labels_stable.csv'

# 加载训练好的模型
def load_model(model_path):
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise ValueError("请更新代码以支持其他类型的模型文件。")

# 实时预测函数
def realtime_predict(model_path, arduino, label_path):
    model = load_model(model_path)
    labels = pd.read_csv(label_path)
    labels_dict = dict(zip(labels['Label'], labels['Grasp Type']))  # 数值标签映射到抓取姿势

    print("模型已加载，正在监听 Arduino 数据...")

    try:
        while True:
            arduino.reset_input_buffer()  # 清空缓冲区，避免读取旧数据
            time.sleep(0.1)  # 小延迟
            data = arduino.readline().decode('utf-8').strip()
            if not data:
                continue  # 跳过空数据

            # 输出原始数据，便于调试
            print("原始 Arduino 数据: {}".format(data))
            
            try:
                # 解析数据，去除空字符串
                data_list = [float(i) for i in data.split(',') if i.strip()]
                # 确保接收到的数据至少为7维（第一列为时间戳）
                if len(data_list) < 7:
                    print("警告: 数据长度不正确 ({}), 跳过该样本".format(len(data_list)))
                    continue  
                # 丢弃第一列（时间戳），只取后6维数据
                data_list = data_list[1:]
                # 确保数据长度为6
                if len(data_list) != 6:
                    print("警告: 数据长度不正确 ({}), 跳过该样本".format(len(data_list)))
                    continue
                # 检查后6维数据中是否有低于50的值
                if any(x < 50 for x in data_list):
                    print("警告: 数据中存在低于50的值, 跳过该样本")
                    continue
                print("解析后的数据: {}".format(data_list))
            except ValueError:
                print("警告: 接收到格式错误的数据: {}".format(data))
                continue

            # 将数据转换为 NumPy 数组，并重塑为 (1,6)
            data_array = np.array(data_list).reshape(1, -1)

            # 进行预测，并获取各类别概率
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_array)[0]
                predicted_label = np.argmax(probabilities)
            else:
                probabilities = None
                predicted_label = model.predict(data_array)[0]

            grasp_type = labels_dict.get(predicted_label, "Unknown Grasp Type")

            # 打印预测结果和所有类别的概率
            print("\n预测抓取类型: {} (标签 {})".format(grasp_type, predicted_label))
            if probabilities is not None:
                print("类别概率分布:")
                for label, prob in zip(labels_dict.keys(), probabilities):
                    grasp = labels_dict.get(label, "Unknown Grasp Type")
                    print("  {} (标签 {}): {:.4f}".format(grasp, label, prob))
            time.sleep(1)  # 小延迟，避免过快输出

    except KeyboardInterrupt:
        print("\n停止实时预测。")
        arduino.close()

# 示例调用：
realtime_predict(model_path, arduino, label_path)
