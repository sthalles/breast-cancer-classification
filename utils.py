# Authors: Thalles, Felipe, Illiana
# Defines a number of helper procedures

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def tf_dataset_to_numpy(tf_dataset):
    # returns a number array from a tensorflow dataset
    y_pred = []
    for _, ys in tf_dataset:
        y_pred.extend(ys.numpy())
    return y_pred

def ensemble_predict(models, dataset):
    # perform prediction for each model and combine the probabilities
    probs = []
    for model in models:
        p = model.predict(dataset)
        probs.append(p)

    probs = np.array(probs)
    # combine the models' predictions
    probs = np.mean(probs, axis=0)
    return probs

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
           # ... and label them with the respective list entries
           xticklabels=["", "", "normal","", "", "", "tumor", "", ""],
           yticklabels=["", "", "normal","", "", "", "tumor", "", ""],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_eval_scores(targets, pred_scores, predictions):
    # compute and display evaluation metrics
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * (precision * recall) / (precision + recall)

    print("---- Metrics ----")
    print("Precision:", precision)
    print("Recall", recall)
    print("F1 Score:", f1score)
    print("AUC:", roc_auc_score(targets, pred_scores))
    print("B. Accuracy:", balanced_accuracy_score(targets, predictions))
