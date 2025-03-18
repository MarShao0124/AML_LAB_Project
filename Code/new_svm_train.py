from load_data import load_data
from plot_confusion import plot_confusion
from sklearn.svm import SVC
import numpy as np
import pickle

# Load data (assumed to be already normalized in load_data)
(X_train, y_train), (X_test, y_test) = load_data('Stable_dataset.csv', label_columns='Labels', test_size=0.2)

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
print("First 5 rows of X_train:")
print(X_train[0:5, :])
print("First 5 labels of y_train:")
print(y_train[0:5])

# Train the SVM classifier
classifier = SVC(kernel='linear', random_state=42, probability=True).fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Save the model so it can be reused in real-time prediction
with open('Model/svm_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Evaluate the classifier (uncomment to display the confusion matrix)
# plot_confusion(y_test, y_pred)
