import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
sys.path.append('./src/')
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader
import torch
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomRotation,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


## self-defined functions and classes
from mymodel.resnet import ResNetMultiLabel
from mymodel.densenet import DenseNetMultiLabel
from mymodel.vision_transformer import ViTMultiLabel
from util.data import read_dataset_from_folder, read_NIH_large, read_CXP, read_CXP_original, read_CXP_original_val_gender
from util.data import collate_fn
from evaluator import draw_roc_auc_curves

print(datetime.now())

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print("We are using device:", device)

data_name = 'NIH' # 'NIH' or 'CXP'
ModelType = 'ResNet50'  # select 'ResNet50','densenet', 'ViT'

## Load data
if data_name == 'NIH':
    root_dir = './NIH-large/'
    split_dir = 'NIH-age-split' # 'NIH-gender-split' # None #'split_random' # default None: use original split; otherwise follow 8:1:1 randomly-split on all lists (using 'split_random')

    # path = './tune-ResNet-on-NIH/'
    # path = './tune-ResNet50-on-NIH/'
    # path = './tune-ResNet50-on-NIH-train-shuffle-0.125val-lr1e-4/'
    # path = './tune-%s-on-NIH-train-shuffle' % ModelType
    # path = './checkpoints/tune-%s-on-NIH-train-w_mask_blend0.0-shuffle-lr1e-04_rot20' % (ModelType)
    # path = './checkpoints/tune-%s-on-NIH-train-w_mask_blend_gau_noise-shuffle-lr1e-04_rot20' % (ModelType)
    path = './checkpoints/tune-%s-on-NIH-train-w_joint_0e+00-shuffle-lr1e-05_rot20_logitreg0.1' % (ModelType)
    if split_dir is None:
        path += '/'
    else:
        path += '_randomsplit/' if split_dir == 'split_random' else '_' + split_dir + '/'

    label_list = list(np.loadtxt(path + 'label_list.txt', dtype='str'))
    print("Loading checkpoint from:", path, flush=True)
    print("label list:", label_list)
    test_ds, class_labels = read_NIH_large(root_dir, label_list=label_list, test_ds_only=True, split_dir=split_dir)
    # train_val_ds, test_ds, class_labels = read_NIH_large(root_dir, label_list=label_list, test_ds_only=False, split_dir=split_dir)
    # test_ds = train_val_ds
elif data_name == 'CXP':
    root_dir = './CXP/CheXpert-v1.0/'
    split_dir = './CXP/split_random/' # './CXP/split_random/'=8:1:1 or './CXP/original_split/'
    path = './tune-%s-on-CXP-train-shuffle-lr1e-05_rot20' % (ModelType)
    path += '_original/' if 'original' in split_dir else '_randomsplit/'
    
    label_list = list(np.loadtxt(path + 'label_list.txt', delimiter='\t', dtype='str'))
    print("Loading checkpoint from:", path, flush=True)
    print("label list:", label_list, flush=True)
    if 'original' in split_dir:
        test_ds, class_labels = read_CXP_original(root_dir, label_list=label_list, test_ds_only=True, split_dir=split_dir)
    else:
        test_ds, class_labels = read_CXP(root_dir, label_list=label_list, test_ds_only=True, split_dir=split_dir)

num_labels = len(class_labels.names)

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_size = 256 if ModelType != 'ViT' else 224

_val_transforms = Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples
test_ds.set_transform(val_transforms)

print(datetime.now())
print("------------------ Load data done -----------------")

## Load model
# get checkpoint model
checkpoint_files = os.listdir(path)
checkpoint_best, epoch_number_best = None, -1
for checkpoint_file in checkpoint_files:
    if 'checkpoint' not in checkpoint_file:
        continue
    epoch_number = int(checkpoint_file.strip().split('_')[-1])
    if epoch_number_best < epoch_number:
        checkpoint_best = checkpoint_file
        epoch_number_best = epoch_number

# define model
if ModelType == 'ResNet50':
    model = ResNetMultiLabel(num_labels).to(device)
elif ModelType == 'densenet':
    model = DenseNetMultiLabel(num_labels).to(device)
elif ModelType == 'ViT':
    model = ViTMultiLabel(num_labels).to(device)

# load model
model.load_state_dict(torch.load(path + checkpoint_best))
model.eval()


## Evaluate
def multi_label_metrics(predictions, labels, threshold=0.5, verbose=1):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    print("pre:", predictions[:10])
    probs = torch.sigmoid(predictions).cpu().numpy()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    print("Check here.............")
    print("y_true:", labels[:10])
    print("prob:", probs[:10])
    print(roc_auc_score(y_true[:10], probs[:10], average = 'micro'))
    print(roc_auc_score(y_true[:10], probs[:10], average = 'macro'))

    for i, name in enumerate(['Atelectasis', 'Effusion']):
        try:
            auc = roc_auc_score(y_true[:, i], probs[:, i])
            print(f"AUC for {name}: {auc:.4f}")
        except ValueError as e:
            print(f"AUC for {name}: Cannot compute ({e})")
    exit(0)

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc_micro = roc_auc_score(y_true, probs, average = 'micro')
    try:
        roc_auc_macro = roc_auc_score(y_true, probs, average = 'macro')
    except ValueError:
        roc_auc_macro = '0.0'
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc_micro': roc_auc_micro,
               'roc_auc_macro': roc_auc_macro,
               'accuracy': accuracy}
    
    # for each label, get the metrics
    roc_auc_each = {}
    for i, each_class in enumerate(class_labels.names):
        if (sum(y_true[:, i]) == len(y_true[:, 1])) or (sum(y_true[:, i]) == 0.0):
            print("check")
            continue    
        roc_auc_each[each_class] = roc_auc_score(y_true[:, i], probs[:, i])

    if data_name == 'CXP':
        selected_class = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'] # the competition evaluation setup
        s = ",".join(['%.4f' % roc_auc_each[x] for x in selected_class])
        s += ",%.4f" % (sum([roc_auc_each[x] for x in selected_class]) * 1.0 / len(selected_class))
        print(",".join(selected_class) + ',mean', '\n', s)
    # all_class_ranked = sorted(class_labels.names)
    all_class_ranked = sorted(roc_auc_each.keys())
    s = ",".join(['%.4f' % roc_auc_each[x] for x in all_class_ranked])
    s += ",%.4f" % (sum(roc_auc_each.values()) * 1.0 / len(all_class_ranked))
    print(",".join(all_class_ranked) + ',mean', '\n', s)
        
    if verbose:
        print(classification_report(y_true=y_true.astype(int), y_pred=y_pred, target_names=class_labels.names))
        print(metrics)
    return metrics

def evaluate(test_dataloader, threshold=0.5, verbose=1, draw_curve=True):
    y_true = torch.tensor([], dtype=torch.long)
    y_pred = torch.tensor([])
    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata['pixel_values'], vdata['labels'].cpu()
            voutputs = model(vinputs).cpu()
            y_pred = torch.cat((y_pred, voutputs), 0)
            y_true = torch.cat((y_true, vlabels), 0)
    if draw_curve:
        draw_roc_auc_curves(y_true.numpy(), y_pred, target_names = class_labels.names, ModelType=ModelType, data_name=data_name)
    return multi_label_metrics(y_pred, y_true.numpy(), threshold=threshold, verbose=verbose)

print("------------------ Starting to evaluate -----------------")
print(datetime.now())
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=256)

print("When thresholds are all 0.5 ---")
evaluate(test_dataloader, threshold=0.5, draw_curve=False)

## Threshold
thresholds = np.loadtxt(path + '/thresholds.txt')

print("When thresholds use the ones give best f1 on validation ---")
evaluate(test_dataloader, threshold=thresholds)
print(datetime.now())

# print("------------------ Evaluating bias and fairness -----------------")
# print(datetime.now())
# if (data_name == 'CXP') and ('original' in split_dir):
#     val_ds, class_labels = read_CXP_original_val_gender(root_dir, label_list=label_list, split_dir=split_dir)
#     test_ds = val_ds
#     test_ds.set_transform(val_transforms)

# test_ds_female = test_ds.filter(lambda example: example['Sex'] == 'Female')
# test_ds_male = test_ds.filter(lambda example: example['Sex'] == 'Male')
# print("The numbers of female and male in CXP-val: %d and %d" % (len(test_ds_female), len(test_ds_male)))

# # %%
# test_dataloader_female = DataLoader(test_ds_female, collate_fn=collate_fn, batch_size=32)
# evaluate(test_dataloader_female, threshold=thresholds, draw_curve=False)

# # %%
# test_dataloader_male = DataLoader(test_ds_male, collate_fn=collate_fn, batch_size=32)
# evaluate(test_dataloader_male, threshold=thresholds, draw_curve=False)
