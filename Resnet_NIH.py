import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from datasets import load_dataset, Image
import numpy as np
from datasets.features import ClassLabel, Sequence
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomRotation,
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
from util.data import read_dataset_from_folder, read_NIH_large, read_CXP, read_CXP_original
from util.data import collate_fn
from mymodel.resnet import ResNetMultiLabel
from mymodel.densenet import DenseNetMultiLabel
from mymodel.vision_transformer import ViTMultiLabel

# # setup for wandb
import wandb

print(datetime.now())

ModelType = 'ResNet50'  # select 'ResNet50','densenet', 'ViT16_base' or 'ViT', 'ViT16_base_swag'

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 256 if 'ViT' not in ModelType else 224

_train_transforms = Compose(
        [
            RandomHorizontalFlip(),
            RandomRotation(20),
            # RandomResizedCrop(img_size, scale=(0.97, 1.03)), # scale +- 3%
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )


# # # load data
data_name = 'NIH' # 'CXP' or 'HIN'

# # root_dir = './NIH-small/sample/'
# # train_val_ds, test_ds, class_labels = read_dataset_from_folder(root_dir)
if data_name == 'NIH':
    root_dir = './NIH-large/'
    split_dir = 'pre-train' # 'split_random/' # default None: use original split; otherwise follow 8:1:1 randomly-split on all lists (using 'split_random')
    train_val_ds, test_ds, class_labels = read_NIH_large(root_dir, split_dir=split_dir)
elif data_name == 'CXP':
    root_dir = './CXP/CheXpert-v1.0/'
    split_dir = './CXP/split_random/' # './CXP/split_random/'=8:1:1 or './CXP/original_split/'
    print(datetime.now(), " ---------- starting to load data from %s ----------" % split_dir)
    if 'original' in split_dir:
        train_ds, val_ds, test_ds, class_labels = read_CXP_original(root_dir, split_dir=split_dir)
    else:
        train_val_ds, test_ds, class_labels = read_CXP(root_dir, split_dir=split_dir)
    print(datetime.now(), " ---------- load data done ----------")

EPOCHS = 10
LR = 1e-5 # 1e-4 # 1e-4 # 5e-5
print("LR:", LR, '; Epochs:', EPOCHS, flush=True)


# path = './tune-ResNet50-on-NIH'
# path = './tune-ResNet50-on-NIH-train-shuffle-0.125val/'
path = './checkpoints/tune-%s-on-%s-train-shuffle-lr%.e_rot20_run2' % (ModelType, data_name, LR)
if split_dir:
    if (data_name == 'CXP') and ('original' in split_dir):
        path += '_original/'
    else:
        path += '_randomsplit/' if split_dir == 'split_random' else '_' + split_dir + '/'
else:
    path += '/'

if not os.path.exists(path):
    os.makedirs(path)
print("Checkpoints stored in: ", path)
np.savetxt(path + 'label_list.txt', class_labels.names, fmt='%s')

if not ((data_name == 'CXP') and ('original' in split_dir)):
    ratio = 0.125
    if split_dir: # make sure ratio of train/val/test split is 8:1:1
        ratio = 0.111
    print("Setting %.2f of training set as validation, default ratio is 0.25" % ratio, flush=True)

    # # train-valid split, value 0.25 for ratio: 6:2:2, value 0.125 for 7:1:2
    train_val_ds_ = train_val_ds.train_test_split(test_size=ratio, seed=42)
    val_ds = train_val_ds_['test']
    train_ds = train_val_ds_['train']

# %% [markdown]
# ### Preprocessing the data
# We will now preprocess the data. The model requires 2 things: pixel_values and labels.
# 
# We will perform data augmentaton on-the-fly using HuggingFace Datasets' set_transform method (docs can be found here). This method is kind of a lazy map: the transform is only applied when examples are accessed. This is convenient for tokenizing or padding text, or augmenting images at training time for example, as we will do here.

# %%

def train_transforms(examples):
    # examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    examples['pixel_values'] = [_train_transforms(image) for image in examples['image']]
    return examples

def val_transforms(examples):
    # examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    examples['pixel_values'] = [_val_transforms(image) for image in examples['image']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)
# %%

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print("We are using device:", device, flush=True)

batch_size = 16 # default: 16
print("Using batch_size: %d, default: 16" % batch_size)
train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=256)

# %%
# batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     print(k, v.shape)
#     if k == 'labels':
#       print(v)

# %% [markdown]
# ### Define the model

# %%

# Instantiate the model
num_labels = len(class_labels.names)
if ModelType == 'ResNet50':
    model = ResNetMultiLabel(num_labels).to(device)
elif ModelType == 'densenet':
    model = DenseNetMultiLabel(num_labels).to(device)
elif 'ViT' in ModelType:
    weight_v = 'base'
    if ModelType == 'ViT16_base_swag':
        weight_v = 'base_swag'
    model = ViTMultiLabel(num_labels, weight_v=weight_v).to(device) 


criterion = nn.BCEWithLogitsLoss()

wandb.init(
    project="SpuriousCorrelation",     # This will create or use an existing project
    # name="experiment-name",          # Optional: a name for this specific run
    config={
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "learning_rate": LR,
        "checkpoint_save_to": path,
        # Add other hyperparameters if desired
    }
)

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

        # wandb logging per batch
        wandb.log({
            'loss/onlycls': loss.item(),
            'epoch': epoch_index,
            'step': epoch_index * len(train_dataloader) + i,
            'lambda': 0.0,
        })

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


best_epoch = -1
best_vloss = 1_000_000.

print(datetime.now(), flush=True)

for epoch in range(EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)

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
    print('LOSS train {} valid {} valid_roc_auc {} valid_f1 {}'.format(avg_loss, avg_vloss, roc_auc / (i+1), f1 / (i+1)), flush=True)
    
    wandb.log({
        "loss/train": avg_loss,
        "loss/val": avg_vloss,
        "val/roc_auc": roc_auc / (i+1),
        "val/f1": f1 / (i+1),
        "epoch": epoch
    })

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    # avg_vloss = - roc_auc / (i+1) # track best model with best auc score
    if avg_vloss < best_vloss:
        if not os.path.exists(path):
            os.makedirs(path)
        best_vloss = avg_vloss
        best_epoch = epoch
        model_path = path + '/checkpoint_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
    
    # # decay lr when no val loss improvement in 3 epochs; break if no val loss improvement in 5 epochs
    # if ((epoch - best_epoch) >= 3):
    #     if avg_vloss > best_vloss:
    #         print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
    #         LR = LR / 2
    #         print("created new optimizer with LR " + str(LR))
    #         if ((epoch - best_epoch) >= 5):
    #             print("no improvement in 5 epochs, break")
    #             break
    # # early stop
    if ((epoch - best_epoch) >= 3):
        print("no improvement in 3 epochs, break")
        break


# %% [markdown]
# ### Evaluate on Testds 
# Consider metrics for multi-class classification
# Load best model for evaluation 
model.eval()
model.load_state_dict(torch.load(model_path))

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
        print(metrics)
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

print(datetime.now())

thresholds = optimize_threshold_metric(model, val_dataloader)
np.savetxt(path + '/thresholds.txt', thresholds)
# np.savetxt(path + '/label_list.txt', class_labels.names, fmt='%s')


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

print(datetime.now())