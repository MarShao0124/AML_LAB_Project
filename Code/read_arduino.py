import serial 
import time
import pickle
import pandas as pd

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
label_path = 'grasp_labels_stable.csv'

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def realtime_predict(model_path, arduino, label_path):
    model = load_model(model_path)
    labels = pd.read_csv(label_path)
    labels = dict(zip(labels['Label'], labels['Grasp Type']))
    while True:
        data = arduino.readline()
        print(data)
        prediction = model.predict(data)
        print(labels[prediction])
        time.sleep(0.1)
    