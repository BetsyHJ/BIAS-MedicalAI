from datasets import load_dataset, Image
import os
import numpy as np
from datasets.features import ClassLabel, Sequence
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

import sys
sys.path.append('./src/')

import warnings
warnings.filterwarnings("ignore")  # TODO: check

## self-defined functions and classes
from util.data import read_dataset_from_folder, read_NIH_large


# root_dir = './NIH-small/sample/'
# train_val_ds, test_ds, class_labels = read_dataset_from_folder(root_dir)
root_dir = './NIH-large/'
train_val_ds, test_ds, class_labels = read_NIH_large(root_dir)

# # train-valid split, ratio: 6:2
train_val_ds_ = train_val_ds.train_test_split(test_size=0.25, seed=42)
val_ds = train_val_ds_['test']
train_ds = train_val_ds_['train']

# %% [markdown]
# ### Preprocessing the data
# We will now preprocess the data. The model requires 2 things: pixel_values and labels.
# 
# We will perform data augmentaton on-the-fly using HuggingFace Datasets' set_transform method (docs can be found here). This method is kind of a lazy map: the transform is only applied when examples are accessed. This is convenient for tokenizing or padding text, or augmenting images at training time for example, as we will do here.

# %%

size = 224

_train_transforms = Compose(
        [
            # RandomResizedCrop(size),
            # RandomHorizontalFlip(),
            Resize(size),
            ToTensor(),
            # normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            # CenterCrop(size),
            ToTensor(),
            # normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

# %%
# from monai.transforms import (
#     Activations,
#     EnsureChannelFirst,
#     AsDiscrete,
#     Compose,
#     LoadImage,
#     RandFlip,
#     RandRotate,
#     RandZoom,
#     ScaleIntensity,
# )
# import torch
# import torch.nn as nn

# # # following: https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb

# _train_transforms = Compose(
#     [
#         RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
#         RandFlip(spatial_axis=0, prob=0.5),
#         RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
#     ]
# )

# _val_transforms = Compose([])

# def train_transforms(examples):
#     examples['pixel_values'] = [_train_transforms(image) for image in examples['image']]
#     return examples

# def val_transforms(examples):
#     examples['pixel_values'] = [_val_transforms(image) for image in examples['image']]
#     return examples

# # Set the transforms
# train_ds.set_transform(train_transforms)
# val_ds.set_transform(val_transforms)
# test_ds.set_transform(val_transforms)

# %%
# from monai.transforms import LoadImageD, EnsureChannelFirstD, ScaleIntensityD, Compose

# transform = Compose(
#     [
#         LoadImageD(keys="image", image_only=True),
#         EnsureChannelFirstD(keys="image"),
#         ScaleIntensityD(keys="image"),
#     ]
# )
# transform(train_ds[0]['image'])

# %%

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    labels = torch.tensor([example["labels"] for example in examples]).to(device).float() # change for one-hot multilabels
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=4)

# %%
batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
    if k == 'labels':
      print(v)

# %% [markdown]
# ### Define the model

# %%

# Define the model
class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiLabel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Instantiate the model
num_labels = len(class_labels.names)
model = ResNetMultiLabel(num_labels).to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        inputs, labels = data['pixel_values'], data['labels']
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # pbar.set_description('  batch {} loss: {}'.format(i + 1, loss.item()))
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            # pbar.set_description('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('logs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    # pbar.set_description('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    roc_auc, f1 = 0.0, 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata['pixel_values'], vdata['labels']
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            vprobs = torch.sigmoid(voutputs).cpu().numpy()
            y_preds = np.zeros(vprobs.shape)
            y_preds[np.where(vprobs >= 0.5)] = 1
            roc_auc += roc_auc_score(vlabels.cpu().numpy(), vprobs, average = 'micro')
            f1 += f1_score(vlabels.cpu().numpy(), y_preds, average = 'micro')
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {} valid_roc_auc {} valid_f1 {}'.format(avg_loss, avg_vloss, roc_auc / len(val_dataloader), f1 / len(val_dataloader)))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        path = './tune-ResNet-on-NIH'
        if not os.path.exists(path):
            os.makedirs(path)
        best_vloss = avg_vloss
        model_path = path + '/checkpoint_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# %% [markdown]
# ### Evaluate on Testds 
# Consider metrics for multi-class classification
# 

# %%
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
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
    # labels = train_ds.features['labels_list']
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(xticks_rotation=45)

    return metrics
        


# %%
## Tuning threshold on val_ds: https://vitaliset.github.io/threshold-dependent-opt/ on f_1 metric
## TODO: we can also use roc_curve to decide the best threshold: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
def optimize_threshold_metric(model, val_dataloader, threshold_grid=None):
    y_true = torch.tensor([], dtype=torch.long)
    predictions = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata['pixel_values'], vdata['labels'].cpu()
            voutputs = model(vinputs).cpu()
            predictions = torch.cat((predictions, voutputs), 0)
            y_true = torch.cat((y_true, vlabels), 0)

    if threshold_grid is None:
        threshold_grid = np.arange(0.01, 1, 0.01)
    optimal_thresholds = np.zeros(num_labels)
    for i in range(num_labels):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in threshold_grid:
            probs = torch.sigmoid(predictions).cpu().numpy()
            # next, use threshold to turn them into integer predictions
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= threshold)] = 1
            f1 = f1_score(y_true[:, i], y_pred[:, i], average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        optimal_thresholds[i] = best_threshold
    print("Optimal thresholds for each category:", optimal_thresholds)
    return optimal_thresholds


thresholds = optimize_threshold_metric(model, val_dataloader)

# %%
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

test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=32)
evaluate(test_dataloader, threshold=thresholds)

# %%
print("When thresholds are all 0.5 ---")
evaluate(test_dataloader, threshold=0.5)

# %% [markdown]
# ### Evaluate on fairness metrics
# E.g., performance difference between female and male groups.

# %%
# test_ds_female = test_ds.filter(lambda example: example['Patient Gender'] == 'F')
# test_ds_male = test_ds.filter(lambda example: example['Patient Gender'] == 'M')

# # %%
# test_dataloader_female = DataLoader(test_ds_female, collate_fn=collate_fn, batch_size=32)
# evaluate(test_dataloader_female, threshold=thresholds)

# # %%
# test_dataloader_male = DataLoader(test_ds_male, collate_fn=collate_fn, batch_size=32)
# evaluate(test_dataloader_male, threshold=thresholds)
