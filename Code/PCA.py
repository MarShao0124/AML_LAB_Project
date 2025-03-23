from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Load data (assumed to be already normalized in load_data)
(X_train, y_train), (X_test, y_test) = load_data('Stable_Dataset.csv', label_columns='Labels', test_size=0.2)

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

# PCA
# Perform PCA
pca = PCA(n_components=3)  # Reduce to 2 dimensions for visualization
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Load grasp labels
labels_df = pd.read_csv('grasp_labels_stable.csv')
label_mapping = dict(zip(labels_df['Label'], labels_df['Grasp Type']))
# Map numerical labels to text labels
y_train_text = [label_mapping[label] for label in y_train]
y_test_text = [label_mapping[label] for label in y_test]

# Plotting the 3D PCA results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', edgecolor='k', s=40)
legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
ax.add_artist(legend1)
ax.set_title('3D PCA of Training Data')
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_zlabel('Third principal component')

# Create a legend with text labels
handles, _ = scatter.legend_elements()
unique_labels = np.unique(y_train)
text_labels = [label_mapping[label] for label in unique_labels]
ax.legend(handles, text_labels, title="Labels")

plt.show()