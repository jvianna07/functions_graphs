from typing import Dict, List, Tuple
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mpl_chord_diagram import chord_diagram

from tensorflow.keras.models import load_model, Model


def predict(model_path: str, x: np.ndarray):

    model: Model = load_model(model_path)

    y_pred: np.ndarray = model.predict(x)

    return y_pred

def compare_predictions(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, labels: Dict[int, str], n: int = None):
    if not n:
        n = randint(0, len(x))
    
    true_label = labels[np.argmax(y[n])]
    pred_label = labels[np.argmax(y_pred[n])]

    plt.imshow(x[n])
    plt.title(f"{n}: Expected '{true_label}', got '{pred_label}'")
    plt.axis("off")
    plt.show()

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    cm: np.ndarray = confusion_matrix(
        y_true=y_true.argmax(axis=1),
        y_pred=y_pred.argmax(axis=1)
    )

    return cm
    
def display_cm(cm: np.ndarray, labels: List[str]):
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    cm_display.plot(cmap='inferno')
    plt.xticks(rotation=90)
    plt.show()

def frame_cm(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(cm)

def frame_errors(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    data_errors = {
        "y_true": y_true.argmax(axis=1),
        "y_pred": y_pred.argmax(axis=1),
    }

    data_errors["y_true_label"] = [labels[i] for i in data_errors["y_true"]]
    data_errors["y_pred_label"] = [labels[i] for i in data_errors["y_pred"]]

    hits = []

    for true, pred in zip(data_errors["y_true"], data_errors["y_pred"]):
        if true == pred:
            hits.append(True)
        else:
            hits.append(False)

    data_errors["hits"] = hits

    df_errors = pd.DataFrame(data_errors)
    df_errors = df_errors.loc[df_errors["hits"] == False]
    df_errors = df_errors.sort_values(by=["y_true", "y_pred"])
    
    return df_errors

def predict_image(image_path: str, model_path: str, labels: List[str]):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)

    model = load_model(model_path)
    y_pred = np.argmax(model.predict(image.reshape(1,128,128,3)))
    y_pred_label = labels[y_pred]

    plt.imshow(image)
    plt.title(y_pred_label)
    plt.axis("off")
    plt.show()

def get_colors(n: int, palette: str = "hls") -> List[str]:

    colors: List[Tuple[float]] = sns.color_palette(palette=palette, n_colors=n)

    colors_rgb: List[List[int]] = []

    for color in colors:
        norm_color: List[int] = [round(x * 255) for x in color]
        colors_rgb.append(norm_color)

    rgb2hex = lambda r, g, b: "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    colors_hex: List[str] = [rgb2hex(*color) for color in colors_rgb]
    
    return colors_hex

def show_chord(cm: np.ndarray, labels: list[str]):
    cm_zeros = cm.copy()

    for i in range(cm_zeros.shape[0]):
        for j in range(cm_zeros.shape[1]):
            if i == j:
                cm_zeros[i, j] = 0

    colors = get_colors(len(labels))

    return chord_diagram(
        cm_zeros,
        labels,
        colors=colors,
        chord_colors=colors,
        width=.07, pad=2, gap=.01,
        rotate_names=True, fontsize=12, show=True
    )

