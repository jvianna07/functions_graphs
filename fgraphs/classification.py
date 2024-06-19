from random import randint
from typing import Dict, Tuple
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.saving import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History


def load_preprocess(preprocess_file: str) -> Dict[str, str]:

    with open(preprocess_file, "rb") as f:
        data = pickle.load(f)

    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = data

    x_train = x_train.astype("float32") / 255
    x_train = np.array([image.reshape((128, 128, 3)) for image in x_train])

    x_validation = x_validation.astype("float32") / 255
    x_validation = np.array([image.reshape((128, 128, 3)) for image in x_validation])

    x_test = x_test.astype("float32") / 255
    x_test = np.array([image.reshape((128, 128, 3)) for image in x_test])

    sets = {
        "x": {
            "train": x_train,
            "validation": x_validation,
            "test": x_test
        },
        "y": {
            "train": y_train,
            "validation": y_validation,
            "test": y_test
        }
    }

    return sets

def show_image(x: np.ndarray, y: np.ndarray, labels: Dict[int, str], n: int = None):
    if not n:
        n = randint(0, len(x))

    y_label = labels[np.argmax(y[n])]

    plt.imshow(x[n])
    plt.title(f"{n}: {y_label}")
    plt.axis("off")
    plt.show()

def build_cnn(input_shape: Tuple,
    conv1_filters: int, conv1_kernel_dim: int, activation_conv1: str, mp1_dim: int,
    conv2_filters: int, conv2_kernel_dim: int, activation_conv2: str, mp2_dim: int,
    conv3_filters: int, conv3_kernel_dim: int, activation_conv3: str,
    dense1_units: int, activation_dense1: str,
    dense2_units: int, activation_dense2: str, lr: float
):
    
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(filters=conv1_filters, kernel_size=(conv1_kernel_dim, conv1_kernel_dim), activation=activation_conv1),
        MaxPooling2D(pool_size=(mp1_dim, mp1_dim)),

        Conv2D(filters=conv2_filters, kernel_size=(conv2_kernel_dim, conv2_kernel_dim), activation=activation_conv2),
        MaxPooling2D(pool_size=(mp2_dim, mp2_dim)),

        Conv2D(filters=conv3_filters, kernel_size=(conv3_kernel_dim, conv3_kernel_dim), activation=activation_conv3),

        Flatten(),

        Dense(dense1_units, activation=activation_dense1),

        Dense(dense2_units, activation=activation_dense2)
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )
    
    return model

def fit(
    model: Model,
    x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray,
    epochs: int, es_patience: int
    ):

    # Definir Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",   # Métrica a ser monitorada
        patience=es_patience, # Número de épocas sem melhora antes de parar o treinamento
        verbose=1             # Mostrar mensagens de log
    )

    # Treinar o modelo
    history = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        validation_data=(x_validation, y_validation),
        callbacks = [early_stopping]
    )

    return history

def plot_history(history: History):
    plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plotar acurácia
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Plotar loss
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation accuracy")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.show()

def save(model: Model, history: History, model_path: str, history_path: str):
    
    save_model(model, model_path)

    with open(history_path, 'w') as f:
        json.dump(history.history, f)


def wrapper_build():
    pass
