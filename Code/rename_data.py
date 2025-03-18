import os

# Define the base directory
base_dir = r"D:\forStudy\AMLlab\ProcessedData_overall"

# List of grasp type subfolders
grasp_folders = [
    "Adduction_Grip",
    "Fixed_hook",
    "Large_diameter",
    "Parallel_extension",
    "Precision_sphere",
    "Sphere_4_finger",
    "Ventral",
    "Writing_Tripod"
]

# Process each grasp folder
for grasp in grasp_folders:
    grasp_path = os.path.join(base_dir, grasp)
    
    if not os.path.isdir(grasp_path):
        print(f"Skipping {grasp_path}, not a directory.")
        continue
    
    # Iterate through subfolders
    for subfolder in os.listdir(grasp_path):
        subfolder_path = os.path.join(grasp_path, subfolder)
        
        # Check if the subfolder name ends with '2'
        if os.path.isdir(subfolder_path) and subfolder.endswith('1'):
            print(f"Processing folder: {subfolder_path}")
            
            # List all files inside the target sub-subfolder
            files = sorted(os.listdir(subfolder_path))  # Sort to maintain order
            
            # Rename files sequentially from 0 to 15
            for index, filename in enumerate(files):
                old_path = os.path.join(subfolder_path, filename)
                file_ext = os.path.splitext(filename)[1]  # Get file extension
                new_filename = f"{index}_new{file_ext}"  # Rename with sequential numbers
                new_path = os.path.join(subfolder_path, new_filename)

                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming {old_path}: {e}")
