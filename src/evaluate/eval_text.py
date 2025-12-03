"""Evaluation utilities for the text model.

Computes MAE/MSE for personality regression and accuracy/F1 for emotions.
Provides simple plotting of predictions vs targets.
"""
import os
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import matplotlib.pyplot as plt


def eval_predictions(traits_true: np.ndarray, traits_pred: np.ndarray, emotions_true: np.ndarray, emotions_pred: np.ndarray):
    """Compute metrics and plot basic visualizations."""
    mae = mean_absolute_error(traits_true, traits_pred)
    mse = mean_squared_error(traits_true, traits_pred)
    acc = accuracy_score(emotions_true, emotions_pred)
    f1 = f1_score(emotions_true, emotions_pred, average="macro")
    print(f"Traits MAE={mae:.4f}, MSE={mse:.4f}")
    print(f"Emotion Acc={acc:.4f}, F1={f1:.4f}")
    # Simple scatter plot for trait 0
    plt.figure()
    plt.scatter(traits_true[:, 0], traits_pred[:, 0])
    plt.xlabel("True Trait O")
    plt.ylabel("Pred Trait O")
    plt.title("Trait O: true vs pred")
    plt.grid(True)
    plt.savefig("eval_trait_o.png")
    print("Saved eval_trait_o.png")
