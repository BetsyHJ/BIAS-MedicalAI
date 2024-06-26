# %%
import os
import numpy as np
import pandas as pd

from transformers import ViTImageProcessor
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
import torch
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# %%

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            # RandomResizedCrop(size),
            # RandomHorizontalFlip(),
            Resize(size),
            # CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            # CenterCrop(size),
            ToTensor(),
            normalize,
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
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    labels = torch.tensor([example["labels"] for example in examples]).to(device).float() # change for one-hot multilabels
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=512)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=512)


# %% [markdown]
# ### Define the model
# Here we define the model. We define a ViTForImageClassification, which places a linear layer (nn.Linear) on top of a pre-trained ViTModel. The linear layer is placed on top of the last hidden state of the [CLS] token, which serves as a good representation of an entire image.
# 
# The model itself is pre-trained on ImageNet-21k, a dataset of 14 million labeled images. You can find all info of the model we are going to use here.
# 
# We also specify the number of output neurons by setting the id2label and label2id mapping, which we be added as attributes to the configuration of the model (which can be accessed as ```model.config```).

# %%
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Load the model and configure it with the number of labels
labels_list = class_labels.names
num_labels = len(labels_list)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_labels,
    id2label = dict(zip(list(range(0, num_labels)), labels_list)),
    label2id = dict(zip(labels_list, list(range(0, num_labels))))
)

# # Print model configuration to verify
# print(model.config)

# %% [markdown]
# ### Visualize the model

metric_name = "f1"

args = TrainingArguments(
    f"fine-tune-ViT-on-NIH",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    y_true = labels
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    # 
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

# Compute weights for each class
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
    
trainer = MultilabelTrainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# %%
trainer.train()

# %% [markdown]
# ### Evaluate on Testds 
# Consider metrics for multi-class classification
# 

# %%
outputs = trainer.predict(test_ds)
print(outputs.metrics)

# %%
## Tuning threshold on val_ds: https://vitaliset.github.io/threshold-dependent-opt/ on f_1 metric
## TODO: we can also use roc_curve to decide the best threshold: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
def optimize_threshold_metric(trainer, val_ds, threshold_grid=None):
    outputs = trainer.predict(val_ds)
    y_true = outputs.label_ids
    predictions = outputs.predictions

    if threshold_grid is None:
        threshold_grid = np.arange(0.01, 1, 0.01)
    optimal_thresholds = np.zeros(num_labels)
    for i in range(num_labels):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in threshold_grid:
            probs = torch.sigmoid(torch.Tensor(predictions)).cpu().numpy()
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


thresholds = optimize_threshold_metric(trainer, val_ds)

    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions)).cpu().numpy()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}

    print(classification_report(y_true=y_true.astype(int), y_pred=y_pred, target_names=class_labels.names))
    # labels = train_ds.features['labels_list']
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot(xticks_rotation=45)

    return metrics

y_true = outputs.label_ids
y_pred = outputs.predictions #.argmax(1)

multi_label_metrics(y_pred, y_true, threshold=thresholds)

# %% [markdown]
# ### Save the best fine-tuned model`

# %%
trainer.save_model()
# %%
# my_model = ViTForImageClassification.from_pretrained("./fine-tune-ViT-on-NIH/")

# # %%
# my_trainer = MultilabelTrainer(
#     my_model,
#     args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
#     tokenizer=processor,
# )

# # %%
# outputs = my_trainer.predict(test_ds)
# print(outputs.metrics)

# y_true = outputs.label_ids
# y_pred = outputs.predictions

# multi_label_metrics(y_pred, y_true)

# %%



