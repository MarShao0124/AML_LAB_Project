from load_data import load_data, load_split_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from itertools import cycle

# Load data (assumed to be already normalized in load_data)
(X_train_1, y_train_1), (X_test_1, y_test_1) = load_split_data('Data/Stable_1.csv')
(X_train_2, y_train_2),(X_test_2, y_test_2) = load_split_data('Data/Stable_2.csv')

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train_1 = X_train_1[:, 1:]
X_test_1 = X_test_1[:, 1:]
X_train_2 = X_train_2[:, 1:]
X_test_2 = X_test_2[:, 1:]

# Load grasp labels
labels_df = pd.read_csv('Data/Stable_label_1.csv')
label_mapping = dict(zip(labels_df['Label'], labels_df['Grasp Type']))
# Apply PCA on the data and visualize the data points on reduced dimensional with labels 
pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train_1)
X_train_second = pca.transform(X_train_2)

# Define a list of distinguishable colors
colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange'])

# Create a dictionary to map each label to a color
label_colors = {label: next(colors) for label in np.unique(y_train_2)}

# Visualize the data with original label y_test_2 in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter plot with labels y_train
for label in np.unique(y_train_2):
    indices = np.where(y_train_2 == label)
    ax.scatter(X_train[indices, 0], X_train[indices, 1], X_train[indices, 2], 
               c=label_colors[label], label=label_mapping[label], alpha=0.7)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Training Data with Labels')
ax.legend()
plt.show()