from medmnist import BreastMNIST
from plot_confusion import plot_history
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical


def NN(dropout_rate=0.2, input_dim=6, num_layers=3, initial_neurons=64):
    model = Sequential()
    model.add(Dense(initial_neurons, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(initial_neurons // 2, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    return model

training_data = BreastMNIST(split='train', download=True, size=28)
test_data = BreastMNIST(split='test', download=True, size=28)
validation_data = BreastMNIST(split='val', download=True, size=28)

X_train = training_data.imgs
y_train = training_data.labels
X_val = validation_data.imgs
y_val = validation_data.labels
X_test = test_data.imgs
y_test = test_data.labels

# Normalize the data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Flatten the images
X_train = X_train.reshape((X_train.shape[0], -1))
X_val = X_val.reshape((X_val.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Create and compile the model
model = NN(input_dim=X_train.shape[1], num_layers=6, initial_neurons=512)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val))

# Plot the training history
plot_history(history)