import serial 
import time
import pickle
import pandas as pd
import numpy as np

arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
model_path = 'Model/svm_model.pkl'
label_path = 'grasp_labels_stable.csv'

def load_model(model_path):
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise ValueError("Please update the code to support other model file formats.")

def realtime_predict(model_path, arduino, label_path):
    model = load_model(model_path)
    labels = pd.read_csv(label_path)
    labels_dict = dict(zip(labels['Label'], labels['Grasp Type'])) # Map numeric labels to grasp types

    print("Model loaded, listening for Arduino data...")

    try:
        while True:
            arduino.reset_input_buffer()  # Clear buffer to avoid reading old data
            time.sleep(0.1)   # Small delay
            data = arduino.readline().decode('utf-8').strip()
            if not data:
                continue  # Skip empty data

            # Print raw data for debugging
            print("Raw Arduino data: {}".format(data))
            
            try:
                # Parse data and remove empty strings
                data_list = [float(i) for i in data.split(',') if i.strip()]
                # Ensure at least 7 values are received (first column is timestamp)
                if len(data_list) < 7:
                    print("Warning: Incorrect data length ({}), skipping sample".format(len(data_list)))
                    continue  
                # Discard the first column (timestamp), keep the last 6 values
                data_list = data_list[1:]
                # Ensure data length is exactly 6
                if len(data_list) != 6:
                    print("Warning: Incorrect data length ({}), skipping sample".format(len(data_list)))
                    continue
                # Check if any value is below 50 in the last 6 dimensions
                if any(x < 50 for x in data_list):
                    print("Warning: Data contains values below 50, skipping sample")
                    continue
                print("Parsed data: {}".format(data_list))
            except ValueError:
                print("Warning: Received incorrectly formatted data: {}".format(data))
                continue

            # Convert data to NumPy array and reshape to (1,6)
            data_array = np.array(data_list).reshape(1, -1)

            # Make prediction and obtain class probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_array)[0]
                predicted_label = np.argmax(probabilities)
            else:
                probabilities = None
                predicted_label = model.predict(data_array)[0]

            grasp_type = labels_dict.get(predicted_label, "Unknown Grasp Type")

            # Print prediction results and probabilities for all classes
            print("\nPredicted Grasp Type: {} (Label {})".format(grasp_type, predicted_label))
            if probabilities is not None:
                print("Class Probability Distribution:")
                for label, prob in zip(labels_dict.keys(), probabilities):
                    grasp = labels_dict.get(label, "Unknown Grasp Type")
                    print("  {} (Label {}): {:.4f}".format(grasp, label, prob))
            time.sleep(1)  # Small delay to prevent excessive output

    except KeyboardInterrupt:
        print("\nReal-time prediction stopped.")
        arduino.close()

realtime_predict(model_path, arduino, label_path)
