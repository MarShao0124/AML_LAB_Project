from load_data import load_data,load_split_data
from plot_confusion import plot_confusion, generate_classification_report, plot_history
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping      

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
# Load data (assumed to be already normalized in load_data)
(X_train_1, y_train_1), (X_test_1, y_test_1) = load_split_data('Data/Stable_1.csv')
(X_train_2, y_train_2),(X_test_2, y_test_2) = load_split_data('Data/Stable_2.csv')

# If the CSV includes a timestamp as the first column, drop it to keep only the 6 feature dimensions.
X_train_1 = X_train_1[:, 1:]
X_test_1 = X_test_1[:, 1:]
X_train_2 = X_train_2[:, 1:]
X_test_2 = X_test_2[:, 1:]

# Train the SVM classifier
#classifier = SVC(kernel='rbf', random_state=42, probability=True).fit(combine_X_train_1, combine_y_train_1)
#y_pred = classifier.predict(combine_X_test)
#y_pred_2 = classifier.predict(X_test_2)

combine_X_train_1 = np.concatenate((X_train_1, X_test_1), axis=0)
combine_y_train_1 = np.concatenate((y_train_1, y_test_1), axis=0)
combine_X_train_2 = np.concatenate((X_train_2, X_test_2), axis=0)
combine_y_train_2 = np.concatenate((y_train_2, y_test_2), axis=0)

# Convert labels to one-hot encoding
y_train_1 = to_categorical(y_train_1)
y_test_1 = to_categorical(y_test_1)
y_train_2 = to_categorical(y_train_2)
y_test_2 = to_categorical(y_test_2)
combine_y_train_1 = to_categorical(combine_y_train_1)
combine_y_train_2 = to_categorical(combine_y_train_2)

"""model trained on 2 participants
model = NN(dropout_rate=0.2, input_dim=6, num_layers=4, initial_neurons=512)
model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['categorical_accuracy'])
history = model.fit(np.concatenate((X_train_1, X_train_2), axis=0), np.concatenate((y_train_1,y_train_2), axis=0), epochs=125, batch_size=32,validation_data=(np.concatenate((X_test_1, X_test_2), axis=0), np.concatenate((y_test_1,y_test_2), axis=0)))
plot_history(history,acc='categorical_accuracy')
"""

#early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
def NN_LOSO(drop_rate, num_layer, initial_neurons, batch_size):
    model = NN(dropout_rate=drop_rate, input_dim=6, num_layers=num_layer, initial_neurons=initial_neurons)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['categorical_accuracy'])

    history_1 = model.fit(combine_X_train_1, combine_y_train_1, epochs=100, batch_size=batch_size,validation_split=0.2, verbose=0)
    plot_history(history_1,acc='categorical_accuracy')
    predict = np.argmax(model.predict(combine_X_train_2), axis=1)
    plot_confusion(np.argmax(combine_y_train_2, axis=1), predict, label_path='Data/Stable_label_1.csv')
    generate_classification_report(np.argmax(combine_y_train_2, axis=1), predict, label_path='Data/Stable_label_1.csv')
    #model.save('Model/NN_model_1.h5')

    
    history_2 = model.fit(combine_X_train_2, combine_y_train_2, epochs=100, batch_size=batch_size,validation_split=0.2, verbose=0)
    plot_history(history_2,acc='categorical_accuracy')
    predict = np.argmax(model.predict(combine_X_train_1), axis=1)
    plot_confusion(np.argmax(combine_y_train_1, axis=1), predict, label_path='Data/Stable_label_1.csv')
    generate_classification_report(np.argmax(combine_y_train_1, axis=1), predict, label_path='Data/Stable_label_1.csv')
    #model.save('Model/NN_model_2.h5')

    return (history_1, history_2)


Dropout_rate = 0.2
Num_layers = 4
Initial_neurons = 512
Batch_size = 8
print(f"({Dropout_rate},{Num_layers},{Initial_neurons},{Batch_size}):", end="")

"""
(0.2,5,512,8):(Accuracy: 0.7695, F1 Score: 0.7668)(Accuracy: 0.7564, F1 Score: 0.7602)
(0.3,5,512,8):(Accuracy: 0.8078, F1 Score: 0.7986)(Accuracy: 0.8304, F1 Score: 0.8351)
(0.4,5,512,8):(Accuracy: 0.7147, F1 Score: 0.6992)(Accuracy: 0.7928, F1 Score: 0.7989)
(0.3,6,1024,8):(Accuracy: 0.7684, F1 Score: 0.7618)(Accuracy: 0.7652, F1 Score: 0.7519)
(0.3,6,512,8):(Accuracy: 0.7551, F1 Score: 0.7466)(Accuracy: 0.7551, F1 Score: 0.7466)
(0.3,5,256,8):(Accuracy: 0.7917, F1 Score: 0.7857)(Accuracy: 0.7199, F1 Score: 0.7339)
"""

history = NN_LOSO(Dropout_rate, Num_layers, Initial_neurons, Batch_size)


"""
histories = []
dropout_rates = np.arange(0, 0.7, 0.1)

for rate in dropout_rates:
    history = NN_LOSO(rate, Num_layers, Initial_neurons, Batch_size)
    histories.append((rate, history))

# Plot the validation accuracy and loss as separate subfigures in the same figure
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
colors = plt.cm.viridis(np.linspace(0, 1, len(dropout_rates)))

for idx, (rate, history) in enumerate(histories):
    color = colors[idx]
    axes[0].plot(history[0].history['val_categorical_accuracy'], linestyle='-', color=color, label=f'Dropout {rate:.1f} Val_Acc')
    axes[0].plot(history[0].history['categorical_accuracy'], linestyle='--', color=color, label=f'Dropout {rate:.1f} Train_Acc')
    axes[1].plot(history[0].history['val_loss'], linestyle='-', color=color, label=f'Dropout {rate:.1f} Val_Loss')
    axes[1].plot(history[0].history['loss'], linestyle='--', color=color, label=f'Dropout {rate:.1f} Train_Loss')

axes[0].set_title('Model Accuracy for Different Dropout Rates')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[1].set_title('Model Loss for Different Dropout Rates')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.tight_layout()
plt.show()
"""



