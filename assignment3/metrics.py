import torch

def get_accuracy(prediction, true):
    n_correct = 0
    for i in range(len(true)):
        if prediction[i] == true[i]:
            n_correct +=1

    return n_correct / len(true)

def get_token_accuracy(prediction, true):
    n_correct = 0
    all_true_tokens = 0
    for i in range(len(true)):
        for j, true_token in enumerate(true[i]):
            try:
                prediction[i][j]
            except:
                break
            
            if true_token == prediction[i][j]:
                n_correct += 1
        all_true_tokens += len(true)

    return n_correct / all_true_tokens