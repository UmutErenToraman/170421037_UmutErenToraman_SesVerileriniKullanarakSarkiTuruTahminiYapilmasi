#%%
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


#%%
data_path2 = "oye.json"
DATA_PATH = "oye_augmented.json"
#%%
def load_data(data_path, data_path2):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    with open(data_path2, "r") as fp2:
        data2 = json.load(fp2)

    X2 = np.array(data2["mfcc"])
    y2 = np.array(data2["labels"])
    
    return X, y, X2, y2
#%%
def prepare_dataset(test_size, validation_size):
    
    X, y, X2, y2 = load_data(DATA_PATH, data_path2)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size)
    _, X_test, _, y_test = train_test_split(X2, y2, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test
#%%
def build_model(input_shape):
    model = keras.Sequential()
    
    # Sequence to sequence layerrrr
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
        
    # output layer
    model.add(keras.layers.Dense(11, activation='softmax'))
    
    return model
#%%
def plot_history(history):
    fig, axs = plt.subplots(2)
    
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    plt.show()
#%%
if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.3, 0.2)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # confusion matrix
    
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    model.summary()
    
    # train model
    history = model.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    
    # plot accuracy/error for training and validation
    plot_history(history)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest Accuracy: ', test_acc)
#%%
