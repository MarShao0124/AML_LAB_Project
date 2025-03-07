from load_data import load_data
from plot_confusion import plot_confusion
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load data
(X_train, y_train), (X_test, y_test) = load_data('Stable_dataset.csv', label_columns='Labels', test_size=0.4)

classifier = SVC(kernel='linear', random_state=42).fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# save model
with open('Model/svm_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# evaluate
plot_confusion(y_test, y_pred)



