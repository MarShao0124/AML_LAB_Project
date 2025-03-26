from load_data import load_data,load_split_data
from plot_confusion import plot_confusion, generate_classification_report, plot_history
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model


def NN(dropout_rate=0.2, input_dim=6, num_layers=3, initial_neurons=64):
    model = Sequential()
    model.add(Dense(initial_neurons, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(initial_neurons // 2, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation='softmax'))
    return model

# Main code
# Load data
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

# Convert labels to one-hot encoding
y_train_1 = to_categorical(y_train_1)
y_test_1 = to_categorical(y_test_1)
y_train_2 = to_categorical(y_train_2)
y_test_2 = to_categorical(y_test_2)
combine_y_train = to_categorical(combine_y_train)
combine_y_test = to_categorical(combine_y_test)




# NN
model = NN(dropout_rate=0.5, input_dim=6, num_layers=4, initial_neurons=512)
model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['categorical_accuracy'])
history = model.fit(X_train_1, y_train_1, epochs=125, batch_size=8,validation_data=(X_test_2, y_test_2))

model_path = 'Model/nn_model.h5'
model = load_model(model_path)


# Plot confusion matrix on unseen data
plot_confusion(np.argmax(y_test_2, axis=1), np.argmax(model.predict(X_test_2), axis=1), label_path='Data/Stable_label_1.csv')
generate_classification_report(np.argmax(y_test_2, axis=1), np.argmax(model.predict(X_test_2), axis=1), label_path='Data/Stable_label_1.csv')

# Plot history
plot_history(history,acc='categorical_accuracy')

#save model
#model.save('Model/nn_model.h5')
#print("Model saved")