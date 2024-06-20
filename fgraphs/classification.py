from random import randint
from typing import Dict, Tuple
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt

from keras_tuner import Hyperband # Add to requirements

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model#, Input
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



# Find best hyperparameters to build CNN model for function graph classification
def build_cnn_model(hp, input_shape:tuple =(128, 128, 3),
                    min_conv_layers=1, max_conv_layers=4,
                    min_filters=4, max_filters=64, filter_step=4,
                    min_dense_units=32, max_dense_units=256, dense_step=8):
    
    # Camada de input
    inputs = Input(shape=input_shape)
    x = inputs

    # Conv2D layers interspersed with MaxPooling2D
    for i in range(hp.Int("conv_layers", min_conv_layers, max_conv_layers, default=3)):
        x = Conv2D (
            filters=hp.Int("filters_" + str(i), min_filters, max_filters, step=filter_step, default=8),
            kernel_size=3,  
            activation="relu",
            padding="same",
        )(x)
    # interspersed with MaxPooling2D layer
        x = MaxPooling2D((2,2))(x)
    
    # End with conv2D layer
    x = Conv2D (
            filters=hp.Int("filters_" + str(i), min_filters, max_filters, step=filter_step, default=8),
            kernel_size=3,  
            activation="relu",
            padding="same",
        )(x)

    # Flatten layer
    x = Flatten()(x)

    # Dense layer with neurons and activation functions variables
    x = Dense(
        units=hp.Int("dense_units", min_dense_units, max_dense_units, step=dense_step, default=128),
        activation=hp.Choice("dense_activation", ["relu", "tanh", "sigmoid"])
    )(x)

    # Output layer
    outputs = Dense(10, activation="softmax")(x)

    # Model creation
    model = Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer = "adam",
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    return model


# Search for the best CNN model
def find_best_cnn_model(x_train: np.ndarray, y_train:np.ndarray,
                        x_val:np.ndarray, y_val:np.ndarray,
                        epochs=5):
    
    # Keras Tuner configuration
    tuner = Hyperband(
        hypermodel=build_cnn_model,
        objective="val_accuracy",
        max_epochs=3,
        overwrite=True,
    )

    # Tuner adjustment
    tuner.search(x_train, y_train,
                 epochs=epochs, validation_data=(x_val, y_val))
    
    return tuner



# OTHER CONVNETS: ResNet50, EfficientNetB0, MobileNetV2
def convNet_classification(convNet:str):
    '''ConvNet (str)- Choose between:
                     ResNet50, EfficientNetB0 or MobileNetV2'''

    # Carregar o modelo ResNet50 sem a camada fully-connected (include_top=False)
    if convNet == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif convNet == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif convNet == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    else:
        raise ValueError (f"ConvNet '{convNet}' not found")
    

    # Adicionar camadas fully-connected personalizadas para classificação
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # Definir modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # Congelar camadas convolucionais do ResNet50
    for layer in base_model.layers:
        layer.trainable = False

    # Compilar modelo
    model.compile(
        optimizer= 'Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    

    return model