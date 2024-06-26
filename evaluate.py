import os
import sys
sys.path.append('./src/')
import numpy as np

from torch.utils.data import DataLoader
import torch
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


## self-defined functions and classes
from mymodel.resnet import ResNetMultiLabel
from util.data import read_dataset_from_folder, read_NIH_large
from util.data import collate_fn

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print("We are using device:", device)

## Load data
root_dir = './NIH-large/'
_, test_ds, class_labels = read_NIH_large(root_dir)
num_labels = len(class_labels.names)

size = 224
_val_transforms = Compose(
        [
            Resize(size),
            # CenterCrop(size),
            ToTensor(),
            # normalize,
        ]
    )
def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples
test_ds.set_transform(val_transforms)

## Load model
mode = 'ResNet' #'ViT' or 'ResNet'
path = './tune-ResNet-on-NIH/'
if mode == 'ResNet':
    checkpoint_files = os.listdir(path)
    checkpoint_best, epoch_number_best = None, -1
    for checkpoint_file in checkpoint_files:
        epoch_number = int(checkpoint_file.strip().split('_')[-1])
        if epoch_number_best < epoch_number:
            checkpoint_best = checkpoint_file
            epoch_number_best = epoch_number
    # define model
    model = ResNetMultiLabel(num_labels).to(device)
    # load model
    model.load_state_dict(torch.load(path + checkpoint_best))
    model.eval()


## Threshold
# path = './tune-ResNet-on-NIH'
# thresholds = np.loadtxt(path + '/thresholds.txt')
thresholds = np.array([0.28, 0.03, 0.03, 0.09, 0.01, 0.02, 0.05, 0.07, 0.02, 0.84, 0.01, 0.21, 0.01, 0.08])

## Evaluate
def multi_label_metrics(predictions, labels, threshold=0.5, verbose=1):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.sigmoid(predictions).cpu().numpy()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
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
    return metrics

def evaluate(test_dataloader, threshold=0.5, verbose=1):
    y_true = torch.tensor([], dtype=torch.long)
    y_pred = torch.tensor([])
    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata['pixel_values'], vdata['labels'].cpu()
            voutputs = model(vinputs).cpu()
            y_pred = torch.cat((y_pred, voutputs), 0)
            y_true = torch.cat((y_true, vlabels), 0)
    return multi_label_metrics(y_pred, y_true.numpy(), threshold=threshold, verbose=verbose)


test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=512)
evaluate(test_dataloader, threshold=thresholds)


print("When thresholds are all 0.5 ---")
evaluate(test_dataloader, threshold=0.5)
