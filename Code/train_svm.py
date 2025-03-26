from load_data import load_data, load_split_data
from plot_confusion import plot_confusion, generate_classification_report
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data (assumed to be already normalized in load_data)
(X_train_1, y_train_1), (X_test_1, y_test_1) = load_split_data('Data/Stable_1.csv')
(X_train_2, y_train_2),(X_test_2, y_test_2) = load_split_data('Data/Stable_2.csv')

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train_1 = X_train_1[:, 1:]
X_test_1 = X_test_1[:, 1:]
X_train_2 = X_train_2[:, 1:]
X_test_2 = X_test_2[:, 1:]

combine_X_train = np.concatenate((X_train_1, X_train_2), axis=0)
combine_y_train = np.concatenate((y_train_1, y_train_2), axis=0)
combine_X_test = np.concatenate((X_test_1, X_test_2), axis=0)
combine_y_test = np.concatenate((y_test_1, y_test_2), axis=0)

# Train the SVM classifier
#classifier = SVC(kernel='rbf', random_state=42, probability=True).fit(combine_X_train, combine_y_train)
#y_pred = classifier.predict(combine_X_test)
#y_pred_2 = classifier.predict(X_test_2)

combine_X_train_1 = np.concatenate((X_train_1, X_test_1), axis=0)
combine_y_train_1 = np.concatenate((y_train_1, y_test_1), axis=0)
combine_X_train_2 = np.concatenate((X_train_2, X_test_2), axis=0)
combine_y_train_2 = np.concatenate((y_train_2, y_test_2), axis=0)

# LOSO-CV: Train on participant 1's data, test on participant 2's data
classifier_1 = SVC(kernel='rbf', random_state=42, probability=True).fit(combine_X_train_1, combine_y_train_1)
y_pred_1 = classifier_1.predict(combine_X_train_2)
plot_confusion(combine_y_train_2, y_pred_1, label_path='Data/Stable_label_1.csv')
generate_classification_report(combine_y_train_2, y_pred_1, label_path='Data/Stable_label_1.csv')

# LOSO-CV: Train on participant 2's data, test on participant 1's data
classifier_2 = SVC(kernel='rbf', random_state=42, probability=True).fit(combine_X_train_2, combine_y_train_2)
y_pred_2 = classifier_2.predict(combine_X_train_1)
plot_confusion(combine_y_train_1, y_pred_2, label_path='Data/Stable_label_2.csv')
generate_classification_report(combine_y_train_1, y_pred_2, label_path='Data/Stable_label_2.csv')

# Save the models so they can be reused in real-time prediction
# with open('Model/svm_model_1.pkl', 'wb') as f:
#     pickle.dump(classifier_1, f)
# with open('Model/svm_model_2.pkl', 'wb') as f:
#     pickle.dump(classifier_2, f)


