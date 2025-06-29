from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt 
from datetime import datetime
import torch
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def draw_roc_auc_curves(y_trues, predictions, target_names, ModelType='ResNet50', data_name='NIH'):
    for i in range(len(target_names)):
        fpr, tpr, thresh = roc_curve(y_trues[:, i], predictions[:,i], pos_label=None) 
        try:
            roc_auc = roc_auc_score(y_trues[:, i], predictions[:,i]) 
        except ValueError:
            roc_auc = 0.0
        plt.plot(fpr, tpr, label='%s (%.4f)' % (target_names[i], roc_auc))
        # print(i, len(fpr), y_trues[:, i].sum())

    # roc curve for tpr = fpr  
    # plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    plt.title('Multiclass (%s-%s) ROC curve' % (data_name, ModelType)) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive rate') 
    plt.legend() 
    plt.savefig('%s-%s-roc-%s.pdf' % (data_name, ModelType, str(datetime.now())), bbox_inches='tight', pad_inches=0)
    plt.close()
    # print("ROC_AUC:", roc_auc_score(y_trues, predictions, average=None))

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5, verbose=1):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.sigmoid(predictions).cpu().numpy()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    print(y_true.shape, y_pred.shape)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, probs, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    if verbose:
        print(classification_report(y_true=y_true.astype(int), y_pred=y_pred, target_names=class_labels.names))
        print(metrics)
    # labels = train_ds.features['labels_list']
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(xticks_rotation=45)

    return metrics

def kl_divergence(alpha, num_classes):
    """KL[Dir(alpha) || Dir(1)] for regularization"""
    beta = torch.ones_like(alpha)  # Dir(1)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)

    lnB = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(S_alpha.squeeze(1))
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1) - torch.lgamma(torch.sum(beta, dim=1))

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1) + lnB - lnB_uni
    return kl