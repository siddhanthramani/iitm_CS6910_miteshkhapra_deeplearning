from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def plot_validation_metrics(list_validation_loss, list_validation_accuracy):
    pass