import os
import pandas as pd

def combine_data():
    data_folder = 'ProcessedData'
    output_file = 'combined_dataset.csv'
    labels_file = 'grasp_labels.csv'
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
                trial_path = os.path.join(object_path, file)
                df = pd.read_csv(trial_path, names=columns, index_col=False, usecols=[0,1,2,3,4,5,6])
                df["Labels"] = i
                combined_dataset = pd.concat([combined_dataset, df], ignore_index=True)
            
        i+=1

    combined_dataset.to_csv(output_file, index=False)
    
    # Save grasp_labels dictionary to CSV using pandas
    grasp_labels_df = pd.DataFrame(list(grasp_labels.items()), columns=['Grasp Type', 'Label'])
    grasp_labels_df.to_csv(labels_file, index=False)
    
    return combined_dataset, grasp_labels

combine_data()