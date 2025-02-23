import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def combine_data(output_file, labels_file):
    """
    Function that extract the data from all the data files and combine them into a single CSV file.
    The function also labels all the data and generates a dictionary of the labels in the form of {grasp_type: label}
    """

    data_folder = 'ProcessedData'
    columns = ['Time in ms', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
    
    combined_dataset = pd.DataFrame(columns=columns)

    grasp_labels = {}

    i = 0

    for file_name in os.listdir(data_folder): #different grasp types
        grasp_type_path = os.path.join(data_folder, file_name) 
        grasp_labels[file_name] = i

        for file_object in os.listdir(grasp_type_path): #different objects
            object_path = os.path.join(grasp_type_path, file_object)
            trial_number = 1
            for file in os.listdir(object_path): #different trials
                trial_path = os.path.join(object_path, file)
                new_trial_path = os.path.join(object_path, f"{trial_number}.csv")
                if trial_path != new_trial_path:
                    os.rename(trial_path, new_trial_path)
                df = pd.read_csv(new_trial_path, names=columns, index_col=False, usecols=[0,1,2,3,4,5,6])
                df["Labels"] = i
                combined_dataset = pd.concat([combined_dataset, df], ignore_index=True)
                trial_number += 1
            
        i+=1

    combined_dataset.to_csv(output_file, index=False)
    
    # Save grasp_labels dictionary to CSV using pandas
    grasp_labels_df = pd.DataFrame(list(grasp_labels.items()), columns=['Grasp Type', 'Label'])
    grasp_labels_df.to_csv(labels_file, index=False)
    


def find_stationary(df, window_size=20, std_threshold=1,stable_window_count=5):
    """
    Finds the first stationary time point for each flex sensor signal using the ADF test.
    Returns a list with the stationary times and corresponding sensor values.
    """

    sensor_cols = df.columns[1:]  # Exclude the time column
    start_indices = []
  

    for col in sensor_cols:
        # calculate the rolling standard deviation of the sensor data
        std_series = df[col].rolling(window=window_size).std()
        meets_condition = std_series < std_threshold
        meets_condition.fillna(False, inplace=True)
        
        # Find the first stable window
        consecutive_count = 0
        start_idx = -1
        for i in range(len(meets_condition)):
            if meets_condition.iloc[i]:
                consecutive_count += 1
                if consecutive_count >= stable_window_count:
                    start_idx = i - stable_window_count + 1
                    break
            else:
                consecutive_count = 0
                
        if start_idx == -1:
            print(f"Sensor {col} does not have a stable window")

        else:
            start_indices.append(start_idx)
    
    # Find the latest start index
    if len(start_indices) == 0:
        print("No stable window found for any sensor")
        return np.array([])
    
    overall_start = max(start_indices)
    
    return df.iloc[overall_start:].values

def extract_stable_dataset(output_file,dict_file):
    data_folder = 'ProcessedData'
    columns = ['Time in ms', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
    
    combined_dataset = pd.DataFrame(columns=columns)

    grasp_labels = {}

    i = 0

    for file_name in os.listdir(data_folder): #different grasp types
        grasp_type_path = os.path.join(data_folder, file_name) 
        grasp_labels[file_name] = i

        for file_object in os.listdir(grasp_type_path): #different objects
            object_path = os.path.join(grasp_type_path, file_object)

            for file in os.listdir(object_path): #different trials
                path = os.path.join(object_path, file)
                df = pd.read_csv(path, names=columns, index_col=False, usecols=[0,1,2,3,4,5,6])
                stationary_data = find_stationary(df)
                if len(stationary_data) == 0:
                    print(f"No stationary data found for {path}")
                else:
                    stationary_data = pd.DataFrame(stationary_data, columns=columns)
                    stationary_data["Labels"] = i
                    combined_dataset = pd.concat([combined_dataset, stationary_data], ignore_index=True)
        i+=1

    combined_dataset.to_csv(output_file, index=False)
    
    # Save grasp_labels dictionary to CSV using pandas
    grasp_labels_df = pd.DataFrame(list(grasp_labels.items()), columns=['Grasp Type', 'Label'])
    grasp_labels_df.to_csv(dict_file, index=False)



def plot_grasp(path,stationary_point=True):
    """
    Function that plots the data of a single grasp type.
    """

    columns = ['Time in ms', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'flex6']
    df = pd.read_csv(path, names=columns, index_col=False, usecols=[0,1,2,3,4,5,6])

    plt.figure()
    plt.plot(df['Time in ms'], df['flex1'], label='flex1')
    plt.plot(df['Time in ms'], df['flex2'], label='flex2')
    plt.plot(df['Time in ms'], df['flex3'], label='flex3')
    plt.plot(df['Time in ms'], df['flex4'], label='flex4')
    plt.plot(df['Time in ms'], df['flex5'], label='flex5')
    plt.plot(df['Time in ms'], df['flex6'], label='flex6')

    if stationary_point:
        stationary_data = find_stationary(df)
        for i in range(1, 7):
            plt.scatter(stationary_data[:, 0], stationary_data[:, i], c='r', s=10)

    plt.xlabel('Time in ms')
    plt.ylabel('Flex Sensor Value')
    plt.title(path)
    plt.xlim(0, 4000)
    plt.ylim(0, 300)
    plt.legend()
    plt.show()


extract_stable_dataset('Stable_dataset.csv', 'grasp_labels_stable.csv')

combine_data('Total_dataset.csv', 'grasp_labels_total.csv')

"""
# Plot the data of a single grasp type
data_folder = 'ProcessedData'
Grasp_type = 'Ventral'
Object = 'stick'
for file in os.listdir(os.path.join(data_folder, Grasp_type, Object)):
    path = os.path.join(data_folder, Grasp_type, Object, file)
    plot_grasp(path)
"""
