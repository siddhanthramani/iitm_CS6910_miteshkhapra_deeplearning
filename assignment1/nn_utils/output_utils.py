from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

def get_accuracy_metrics(y_true, y_pred, micron_on=0,
                        macro_on=0, weighted_on=0, confusion_on=0):
    return_values = []
    accuracy = accuracy_score(y_true, y_pred)
    return_values.append(accuracy)

    if micron_on:
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return_values.append(precision_micro)
        return_values.append(recall_micro)
        return_values.append(f1_micro)

    if macro_on:
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return_values.append(precision_macro)
        return_values.append(recall_macro)
        return_values.append(f1_macro)

    if weighted_on:
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        return_values.append(precision_weighted)
        return_values.append(recall_weighted)
        return_values.append(f1_weighted)

    if confusion_on:
        confusion = confusion_matrix(y_true, y_pred)
        return_values.append(confusion)
    
    return return_values

def plot_confusion_matrix(wandb, y_true, y_pred):
    df = pd.DataFrame({"y_true" : y_true, "y_pred" : y_pred})
    a = df.groupby(["y_true", "y_pred"]).agg(count=("y_pred", "count")).reset_index()
    b = df.groupby("y_true").agg(total=("y_true", "count")).reset_index()
    c = pd.merge(a, b, on="y_true")
    c["percent"] = c["count"]/c["total"]
    c[["y_true", "y_pred", "percent"]].sort_values(["y_true", "percent"], ascending=[True, False])
    plt.scatter(c["y_true"], c["y_pred"], s=c["percent"]*10)
    wandb.log({"confusion matrix": [wandb.Image(plt)]})
