from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt 
from datetime import datetime

def draw_roc_auc_curves(y_trues, predictions, target_names):
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
    plt.title('Multiclass (NIH-ResNet) ROC curve') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive rate') 
    plt.legend() 
    plt.savefig('NIH-ResNet-roc-%s.pdf' % str(datetime.now()), bbox_inches='tight', pad_inches=0)
    plt.close()
    # print("ROC_AUC:", roc_auc_score(y_trues, predictions, average=None))