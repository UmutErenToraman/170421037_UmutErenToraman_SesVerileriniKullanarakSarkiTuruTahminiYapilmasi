#%%
import json
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from keras.src.utils import to_categorical

#%%
data_path_train = "oye_augmented.json"  # Train verisi
data_path_test = "oye.json"  # Test verisi
#%%
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y
#%%
def prepare_dataset(validation_size):

    X_train, y_train = load_data(data_path_train)  # Augmented dataset
    X_test, y_test = load_data(data_path_test)  # Orijinal dataset

    # Validation set oluşturma
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    print("Unique labels train:", np.unique(y_train))
    print("Unique labels val:", np.unique(y_validation))
    print("Unique labels test:", np.unique(y_test))
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)  # Etiketleri düzelt
    y_validation = encoder.transform(y_validation)
    y_test = encoder.transform(y_test)  # Aynı dönüşümü test setine uygula

    print("after encoder")
    print("Unique labels train:", np.unique(y_train))
    print("Unique labels val:", np.unique(y_validation))
    print("Unique labels test:", np.unique(y_test))


    return X_train, X_validation, X_test, y_train, y_validation, y_test
#%%
def build_model(input_shape, num_classes):
    model = keras.Sequential()
    
    # Sequence to sequence layerrrr
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
        
    # output layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
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
def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
#%%
def calculate_f1_score(model, X_test, y_test):
    # Modelin tahminlerini al
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Softmax çıktılarını sınıf indeksine çevir

    # Confusion matrix oluştur
    cm = confusion_matrix(y_test, y_pred)

    # F1-score’u hesapla
    f1_scores = f1_score(y_test, y_pred, average=None)  # Her sınıf için ayrı F1-score

    return cm, f1_scores
#%%
def plot_f1_scores(f1_scores, class_labels):
    """ F1-score'ları çubuk grafikte gösterir. """

    plt.figure(figsize=(10, 5))

    # Check if the length of f1_scores matches the length of class_labels
    if len(f1_scores) != len(class_labels):
        print("Error: Length of f1_scores and class_labels must be the same.")
        return  # Exit the function if lengths don't match

    sns.barplot(x=class_labels, y=f1_scores, palette="viridis")

    plt.xlabel("Sınıf")
    plt.ylabel("F1-Score")
    plt.title("Sınıf Bazında F1-Skorları")
    plt.ylim(0, 1)  # F1-score 0 ile 1 arasında olmalı
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
#%%
def calculate_recall(model, X_test, y_test):
    """ Modelin test setindeki tahminlerine göre recall hesaplar. """

    # Modelin tahminlerini al
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Softmax çıktılarını sınıf indeksine çevir

    # Recall hesapla
    recall_scores = recall_score(y_test, y_pred, average=None)  # Her sınıf için ayrı recall

    return recall_scores
#%%
def plot_recall_scores(recall_scores):
    """ Recall skorlarını çubuk grafikte gösterir. """

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(range(len(recall_scores))), y=recall_scores, palette="magma")

    plt.xlabel("Sınıf")
    plt.ylabel("Recall")
    plt.title("Sınıf Bazında Recall Skorları")
    plt.ylim(0, 1)  # Recall 0 ile 1 arasında olmalı
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
#%%
if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.2)


    # Assuming y_train, y_validation, y_test are your integer-encoded labels

    num_classes = len(np.unique(y_train)) # Get the number of unique classes

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)

    # confusion matrix
    
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    model.summary()
    
    # train model
    history = model.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    #history = model.fit(x=X_train, y=y_train_one_hot, validation_data=(X_validation, y_validation_one_hot), batch_size=32, epochs=30)
    
    # plot accuracy/error for training and validation
    plot_history(history)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest Accuracy: ', test_acc)

#%%
print("Unique labels in training set:", np.unique(y_train))
print("Unique labels in test set:", np.unique(y_test))
#%%
# Modelin tahminleri
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
class_labels = [str(i) for i in range(10)]  # 0-10 arası sınıflar varsayıldı
plot_confusion_matrix(y_test, y_pred, class_labels)

# Classification Report
print(classification_report(y_test, y_pred, target_names=class_labels))

# Test sonrası
cm, f1_scores = calculate_f1_score(model, X_test, y_test)

# F1-score plot
plot_f1_scores(f1_scores, class_labels)


# Recall hesapla
recall_scores = calculate_recall(model, X_test, y_test)

# Recall görselleştir
plot_recall_scores(recall_scores)
#%%
