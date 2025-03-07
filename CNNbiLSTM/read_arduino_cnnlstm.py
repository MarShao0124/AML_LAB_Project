import serial 
import time
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque  # Used for maintaining a sliding window

# Initialize serial connection
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
model_path = 'CNNbiLSTM/cnn_lstm_model.pkl'  # Ensure the path is correct
label_path = 'grasp_labels_stable.csv'
WINDOW_SIZE = 10  # Ensure this matches the model training window size

# Load the trained TensorFlow/Keras model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Maintain a sliding window buffer to store the last 10 frames
buffer = deque(maxlen=WINDOW_SIZE)

# Real-time prediction function
def realtime_predict(model_path, arduino, label_path):
    model = load_model(model_path)
    labels = pd.read_csv(label_path)
    labels_dict = dict(zip(labels['Label'], labels['Grasp Type']))

    print("Model loaded. Listening for Arduino data...")

    try:
        while True:
            arduino.reset_input_buffer()  # Clear buffer to avoid reading old data
            time.sleep(0.1)  # Small delay to reduce data loss
            
            data = arduino.readline().decode('utf-8').strip()
            if not data:
                continue  # Skip empty data
            
            # Print raw data for debugging
            print("Raw Arduino data: {}".format(data))
            
            try:
                # Parse the data, skipping any empty elements
                data_list = [float(i) for i in data.split(',') if i.strip()]
                
                # Check if the data length is exactly 6
                if len(data_list) != 6:
                    print("Warning: Incorrect data length ({}), skipping this sample".format(len(data_list)))
                    continue
                
                print("Parsed data: {}".format(data_list))
            except ValueError:
                print("Warning: Malformed data received: {}".format(data))
                continue

            # Append the parsed data to the sliding window buffer
            buffer.append(data_list)

            # Only perform prediction when the buffer is full (contains 10 frames)
            if len(buffer) < WINDOW_SIZE:
                continue

            # Convert the buffer to a NumPy array with shape (1, 10, 6)
            data_array = np.array(buffer).reshape(1, WINDOW_SIZE, 6)
            
            # Make prediction and obtain class probabilities
            probabilities = model.predict(data_array)[0]
            predicted_label = np.argmax(probabilities)
            grasp_type = labels_dict.get(predicted_label, "Unknown Grasp Type")

            # Print prediction results and probabilities for all classes
            print("\nPredicted Grasp Type: {} (Label {})".format(grasp_type, predicted_label))
            print("Class Probabilities:")
            for label, prob in enumerate(probabilities):
                grasp = labels_dict.get(label, "Unknown Grasp Type")
                print("  {} (Label {}): {:.4f}".format(grasp, label, prob))

            time.sleep(1)  # Small delay for stability 

    except KeyboardInterrupt:
        print("\nStopping real-time prediction.")
        arduino.close()

# Example usage:
realtime_predict(model_path, arduino, label_path)
