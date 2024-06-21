from typing import Dict, List, Tuple
from random import randint
from math import sqrt

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

def process_binary_confusion_matrices(cm: np.ndarray, labels: List[str]) -> Dict[str, pd.DataFrame]:
    binary_cms: Dict[str, pd.DataFrame] = {}

    df_cm = pd.DataFrame(cm)

    for i, label in enumerate(labels):

        TP = df_cm.iloc[i, i]
        FP = df_cm.iloc[:, i].sum().sum() - TP
        FN = df_cm.iloc[i, :].sum().sum() - TP
        TN = df_cm.sum().sum() - TP - FP - FN

        # construct DataFrame for current label
        binary_cms[label] = pd.DataFrame(
            data={
                "PREDICTED POSITIVE": [TP, FP],
                "PREDICTED NEGATIVE": [FN, TN]
            },
            index=["ACTUAL POSITIVE", "ACTUAL NEGATIVE"]
        )
    return binary_cms

def report(binary_cms: Dict[str, pd.DataFrame], labels: List[str]):

    report_data: Dict = {
        "label": [label for label in labels],
        "P": [],
        "N": [],
        "TP": [],
        "FN": [],
        "FP": [],
        "TN": [],
        "TPR": [],
        "TNR": [],
        "PPV": [],
        "NPV": [],
        "FNR": [],
        "FPR": [],
        "FDR": [],
        "FOR": [],
        "PLR": [],
        "NLR": [],
        "PT": [],
        "TS": [],
        "PRE": [],
        "ACC": [],
        "BA": [],
        "F1": [],
        "MCC": [],
        "FM": [],
        "BM": [],
        "MK": [],
        "DOR": []
    }

    for i, label in enumerate(labels):

        # extract binary confusion matrix for single label
        binary_cm: pd.DataFrame = binary_cms[label]

        # calculate metrics
        P: int = binary_cm.loc["ACTUAL POSITIVE"].sum()
        N: int = binary_cm.loc["ACTUAL NEGATIVE"].sum()
        TP: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED POSITIVE"]
        FN: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED NEGATIVE"]
        FP: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED POSITIVE"]
        TN: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED NEGATIVE"]
        TPR: float = TP / P if P != 0 else np.nan
        TNR: float = TN / N if N != 0 else np.nan
        PPV: float = TP / (TP + FP) if (TP + FP) != 0 else np.nan
        NPV: float = TN / (TN + FN) if (TN + FN) != 0 else np.nan
        FNR: float = FN / P if P != 0 else np.nan
        FPR: float = FP / N if N != 0 else np.nan
        FDR: float = FP / (FP + TP) if (FP + TP) != 0 else np.nan
        FOR: float = FN / (FN + TN) if (FN + TN) != 0 else np.nan
        PLR: float = TPR / FPR if FPR != 0 else np.nan
        NLR: float = FNR / TNR if TNR != 0 else np.nan
        PT: float = sqrt(FPR) / (sqrt(TPR) + sqrt(FPR)) if (sqrt(TPR) + sqrt(FPR)) != 0 else np.nan
        TS: float = TP / (TP + FN + FP) if (TP + FN + FP) != 0 else np.nan
        PRE: float = P / (P + N) if (P + N) != 0 else np.nan
        ACC: float = (TP + TN) / (P + N) if (P + N) != 0 else np.nan
        BA: float = (TPR + TNR) / 2
        F1: float = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else np.nan
        MCC: float = sqrt(PPV * TPR * TNR * NPV) - sqrt(FDR * FNR * FPR * FOR)
        FM: float = sqrt(PPV * TPR)
        BM: float = TPR + TNR - 1
        MK: float = PPV + NPV - 1
        DOR: float = PLR / NLR if NLR != 0 else np.nan

        # structure metrics in a dictionary
        metrics_current_label = {
            "P": P,
            "N": N,
            "TP": TP,
            "FN": FN,
            "FP": FP,
            "TN": TN,
            "TPR": TPR,
            "TNR": TNR,
            "PPV": PPV,
            "NPV": NPV,
            "FNR": FNR,
            "FPR": FPR,
            "FDR": FDR,
            "FOR": FOR,
            "PLR": PLR,
            "NLR": NLR,
            "PT": PT,
            "TS": TS,
            "PRE": PRE,
            "ACC": ACC,
            "BA": BA,
            "F1": F1,
            "MCC": MCC,
            "FM": FM,
            "BM": BM,
            "MK": MK,
            "DOR": DOR
        }

        # update general dictionary of metrics
        for key in report_data:
            if key != "label":
                report_data[key].append(metrics_current_label[key])

    # turn general dictionary of metrics into a dataframe
    return pd.DataFrame(report_data)



# Show Best hyperparameters for function graphs problem
def show_best_hps (tuner_hp):
    best_hps_4_fgraphs = tuner_hp.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
          
    Best hyperparameters are:
    - Nr of convolutional layers: {best_hps_4_fgraphs.get('conv_layers')}
    - Filters in convolutional layers: {[best_hps_4_fgraphs.get('filters_' + str(i)) for i in range(best_hps_4_fgraphs.get('conv_layers'))]}
    - Nr of neurons in dense layer: {best_hps_4_fgraphs.get('dense_units')}
    - Activation function in dense layer: {best_hps_4_fgraphs.get('dense_activation')}
    """) 
