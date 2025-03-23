from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical


def NN(dropout_rate=0.2, input_dim=6, num_layers=3, initial_neurons=64):
    model = Sequential()
    model.add(Dense(initial_neurons, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(initial_neurons // 2, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation='softmax'))
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')

    plt.show()




# Main code
# Load data
(X_train_1, y_train_1), (X_test_1, y_test_1) = load_data('Data/Stable_1.csv', label_columns='Labels', test_size=0.2)
(X_train_2, y_train_2), (X_test_2, y_test_2) = load_data('Data/Stable_2.csv', label_columns='Labels', test_size=0)

X_train = X_train_1
X_test = X_test_1
y_train = y_train_1
y_test = y_test_1

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

# Load grasp labels
labels_df = pd.read_csv('Data/Stable_label_1.csv')
label_mapping = dict(zip(labels_df['Label'], labels_df['Grasp Type']))
# Map numerical labels to text labels
y_train_text = [label_mapping[label] for label in y_train]
y_test_text = [label_mapping[label] for label in y_test]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train_2 = to_categorical(y_train_2)


# NN
model = NN(dropout_rate=0, input_dim=X_train.shape[1], num_layers=3, initial_neurons=64)
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


score = model.evaluate(X_train_2[:,1:], y_train_2)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

print(X_train_2.shape)
print(X_train.shape)


# Plot history
plot_history(history)

