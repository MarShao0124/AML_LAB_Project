import serial 
import time
import pickle
import pandas as pd
import numpy as np

# Initialize serial connection
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
model_path = 'Model/svm_model.pkl'
label_path = 'grasp_labels_stable.csv'

# Load the trained model
def load_model(model_path):
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise ValueError("Update the code to read different model file types.")

# Real-time prediction function
def realtime_predict(model_path, arduino, label_path):
    model = load_model(model_path)
    
    # Load labels into a dictionary
    labels = pd.read_csv(label_path)
    labels_dict = dict(zip(labels['Label'], labels['Grasp Type']))

    print("Model loaded. Listening for Arduino data...")

    try:
        while True:
            data = arduino.readline().decode('utf-8').strip()
            if not data:
                continue  # Skip if no data received
            
            try:
                data_list = [float(i) for i in data.split(',')]  # Ensure numerical conversion
            except ValueError:
                print("Warning: Received malformed data:", data)
                continue  # Skip invalid input

            # Convert input into a NumPy array (reshape for model compatibility)
            data_array = np.array(data_list).reshape(1, -1)

            # Make prediction
            prediction = model.predict(data_array)
            
            # Convert prediction to integer
            predicted_label = int(prediction[0])
            
            # Get grasp type
            grasp_type = labels_dict.get(predicted_label, "Unknown Grasp Type")
            print(f"Predicted Grasp Type: {grasp_type}")

            time.sleep(0.5)  # Small delay for stability

    except KeyboardInterrupt:
        print("\nStopping real-time prediction.")
        arduino.close()

# Example usage:
# realtime_predict('your_model.pkl', arduino, label_path)
