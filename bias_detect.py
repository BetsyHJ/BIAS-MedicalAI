# automaticly detect spurious correlation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from datasets import disable_progress_bar
disable_progress_bar()
from datasets import concatenate_datasets

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
from datasets import Dataset, concatenate_datasets

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

## self-defined functions and classes
from mymodel.resnet import ResNetMultiLabel
from mymodel.densenet import DenseNetMultiLabel
from mymodel.vision_transformer import ViTMultiLabel
from util.data import read_dataset_from_folder, read_NIH_large, read_CXP, read_CXP_original, read_CXP_original_val_gender
from util.data import collate_fn
from evaluator import draw_roc_auc_curves, multi_label_metrics

from sklearn.cluster import KMeans

from visualization import visualize_resnet_feature_maps, visualize_resnet_activation_heatmaps, draw_bbox
from visualization import visualize_resnet_activation_heatmaps_batch
print(datetime.now())

# import torchvision.models as models
# resnet = models.resnet18(pretrained=True)
# print(resnet)
# exit(0)

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
    split_dir = 'split_random' # default None: use original split; otherwise follow 8:1:1 randomly-split on all lists (using 'split_random')
    path = './checkpoints/tune-%s-on-NIH-train-w_mask_blend_gau_noise-shuffle-lr1e-04_rot20' % (ModelType)
    path += '_randomsplit/' if split_dir else '/'

    label_list = list(np.loadtxt(path + 'label_list.txt', dtype='str'))
    print("Loading checkpoint from:", path, flush=True)
    print("label list:", label_list)
    train_val_ds, test_ds, class_labels = read_NIH_large(root_dir, label_list=label_list, test_ds_only=False, split_dir=split_dir)
elif data_name == 'CXP':
    root_dir = './CXP/CheXpert-v1.0/'
    split_dir = './CXP/split_random/' # './CXP/split_random/'=8:1:1 or './CXP/original_split/'
    path = './checkpoints/tune-%s-on-CXP-train-shuffle-lr1e-05_rot20' % (ModelType)
    path += '_original/' if 'original' in split_dir else '_randomsplit/'
    label_list = list(np.loadtxt(path + 'label_list.txt', delimiter='\t', dtype='str'))
    print("Loading checkpoint from:", path, flush=True)
    print("label list:", label_list, flush=True)
    if 'original' in split_dir:
        train_ds, val_ds, test_ds, class_labels = read_CXP_original(root_dir, label_list=label_list, test_ds_only=False, split_dir=split_dir)
        train_val_ds = concatenate_datasets([train_ds, val_ds])
    else:
        train_val_ds, test_ds, class_labels = read_CXP(root_dir, label_list=label_list, test_ds_only=False, split_dir=split_dir)

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

# # Check: first try a subset from train_val_ds
# train_val_ds = train_val_ds.train_test_split(test_size=0.01, seed=42)['test']
# print("!!!!!!!!!!!! Try on %d samples from train_val set !!!!!!!!!!!!" % len(train_val_ds))

# all_sample_labels = np.array(train_val_ds[:]['labels'])
# print(all_sample_labels.shape)
# for i in range(len(label_list)):
#     print(label_list[i], np.mean(all_sample_labels[:, i] == 1.0))
# print("no findings:", np.mean(all_sample_labels.sum(1) <= 0.0))
# print(all_sample_labels.sum(1))
# exit(0)

# # here we can use the bbox_list_2017.csv
bbox_info = np.loadtxt("./NIH-large/BBox_List_2017.csv", dtype=str, skiprows=1, delimiter=',') # # Image Index,Finding Label,Bbox [x,y,w,h],,,
# print("----------------- Load testset for BBox List only ------------------")
# bbox_train_split = np.loadtxt('./NIH-large/BBox_List_train_split.csv', dtype=str)
# bbox_info = bbox_info[np.where(bbox_train_split == 'True', False, True)]
# print("----------------- Load %d testset done ---------------" % len(bbox_info))

bbox_dict = {}
duplications = 0
for i in range(len(bbox_info)):
    v = bbox_info[i]
    if v[1] == "Infiltrate":
        v[1] = "Infiltration"
    if v[0] not in bbox_dict:
        bbox_dict[v[0]] = [[v[1], [int(float(x)) for x in v[2:6]]]]
    else:
        bbox_dict[v[0]].append([v[1], [int(float(x)) for x in v[2:6]]])
        duplications += 1
        # print("Duplicate:", v[0], len(bbox_dict[v[0]]))
bbox_image_index = bbox_dict.keys()
name_to_index = {name: i for i, name in enumerate(train_val_ds["Image Index"])}
subset_indices = [name_to_index[name] for name in bbox_image_index if name in name_to_index]
subset = train_val_ds.select(subset_indices)
# train_val_ds = subset
# # some bbox images are in testset 
name_to_index = {name: i for i, name in enumerate(test_ds["Image Index"])}
subset_indices = [name_to_index[name] for name in bbox_image_index if name in name_to_index]
subset2 = test_ds.select(subset_indices)

print("!!!!!!!!!!!!!!!!!!!! Taking all images with BBox even when they are in the testset")
train_val_ds = concatenate_datasets([subset, subset2])



# # add BBox info to those images
class_label2idx = dict(zip(class_labels.names, np.arange(len(class_labels.names))))
# def add_extra(example):
#     image_indices = example["Image Index"]
#     v = [bbox_dict[x] for x in image_indices]
#     return {"BBox": [x[1] for x in v], "target_class": [class_label2idx[x[0]] for x in v]}
# train_val_ds = train_val_ds.map(add_extra, batched=True)
def expand_annotations(example_batch):
    new_batch = {k: [] for k in example_batch.keys()}
    new_batch["BBox"] = []
    new_batch["target_class"] = []
    
    for i, img_idx in enumerate(example_batch["Image Index"]):
        for class_, bbox in bbox_dict[img_idx]:  
            for k in example_batch.keys():
                new_batch[k].append(example_batch[k][i])
            new_batch["BBox"].append(bbox)
            new_batch["target_class"].append(class_label2idx[class_])  # int
    return new_batch
train_val_ds = train_val_ds.map(expand_annotations, batched=True, num_proc=4)
print("Only take the image in the BBox_List file:", len(train_val_ds), ". There are %d duplications" % duplications, flush=True)

# # # get the images with bbox appearing in random-split testset and make sure the splitting putting them in the test
# print("Nr. Unique images in the bbox list", len(bbox_image_index), flush=True) # =880
# image_bbox_in_test = [name for name in bbox_image_index if name not in name_to_index]
# image_bbox_other = [name for name in bbox_image_index if name in name_to_index]
# import random
# selected_bbox_test = random.sample(image_bbox_other, int(len(bbox_image_index) * 0.2) - len(image_bbox_in_test))
# print(len(image_bbox_in_test), len(selected_bbox_test)) # 68, 108
# image_bbox_in_test += selected_bbox_test
# print("Final:", len(image_bbox_in_test)) # 176
# split_test_bbox = []
# for i in range(len(bbox_info)):
#     v = bbox_info[i]
#     if v[0] in image_bbox_in_test:
#         split_test_bbox.append("False")
#     else:
#         split_test_bbox.append("True")
# np.savetxt('./NIH-large/BBox_List_train_split.csv', np.array(split_test_bbox), fmt='%s', delimiter='\n')


train_val_ds.set_transform(val_transforms)

print(datetime.now())
print("------------------ Load data done -----------------", flush=True)

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
    feature_extractor = create_feature_extractor(model, {'resnet.avgpool':'features'})
    feature_shape = 2048
elif ModelType == 'densenet':
    model = DenseNetMultiLabel(num_labels).to(device)
elif ModelType == 'ViT':
    model = ViTMultiLabel(num_labels).to(device)

# load model
model.load_state_dict(torch.load(path + checkpoint_best))
model.eval()

# print(model)

assert ModelType == 'ResNet50'


print("------------------ Visualizing feature map -----------------", flush=True)


# # the below code is not for NIH
# data_entry_info = np.loadtxt("./NIH-large/Data_Entry_2017.csv", dtype=str, skiprows=1, usecols=(0, 7, 8), delimiter=',')
# for i in range(len(data_entry_info)):
#     if data_entry_info[i][0] in bbox_image_index:
#         if bbox_dict[data_entry_info[i][0]][2] == 1: # it has been resized 
#             print("This should not happen!!!!!!!!!!!")
#             continue 
#         OW, OH = float(data_entry_info[i][1]), float(data_entry_info[i][2])
#         weights = 1024.0 / max(OW, OH)
#         bbox_dict[data_entry_info[i][0]][1] = [int(x * weights)for x in bbox_dict[data_entry_info[i][0]][1]]
#         bbox_dict[data_entry_info[i][0]][2] += 1

# # for all possible images in train_val_ds
# print(train_val_ds)
# count_Pneumonia_images = 0
# results = {"hit":[], "iou":[], "dice":[]} # hit, iou, dice
# for i in range(len(train_val_ds)):
#     if train_val_ds[i]['Image Index'] not in bbox_image_index:
#         continue
#     if count_Pneumonia_images >= 100:
#         exit(0)
#     # draw_bbox(train_val_ds[i]['image'], bbox_dict[train_val_ds[i]['Image Index']][1])
#     # print("here", train_val_ds[i], flush=True)
#     input_tensor = train_val_ds[i]['pixel_values'].to(device) # [3, 256, 256]
#     target_classes = list(np.where(np.array(train_val_ds[i]['labels']) == 1.0)[0])
#     selected_classes_label = bbox_dict[train_val_ds[i]['Image Index']][0] # "Pneumonia" # "Atelectasis"
#     if sum(target_classes) >= 1.0:
#         if selected_classes_label not in train_val_ds[i]['Finding Labels']:
#             continue
#         target_classes = np.where(np.array(class_labels.names) == selected_classes_label)[0]
#         # print(train_val_ds[i]['Image Index'], train_val_ds[i]['Finding Labels'], target_classes)
#         print(target_classes, class_labels.names, flush=True)
#         result = visualize_resnet_activation_heatmaps(model, input_tensor, target_classes, image_id=train_val_ds[i]['Image Index'], selected_feature_ids=None, target_class_labels=np.array(class_labels.names)[target_classes], bbox_info=bbox_dict[train_val_ds[i]['Image Index']], ori_img=train_val_ds[i]['image'], save_fig=False)
#         print(result)
#         if result is not None:
#             results["hit"].append(result[0])
#             results["iou"].append(result[1])
#             results["dice"].append(result[2])
#         count_Pneumonia_images += 1
#         continue
# # print(input_tensor.shape)
# # visualize_resnet_feature_maps(model, input_tensor, selected_feature_ids=None) # 
# # visualize_resnet_activation_heatmaps(model, input_tensor, target_classes, selected_feature_ids=None)

# # only look at images w/ BBox
# check if one image just appears once in the BBox file.

print(train_val_ds)
print("Starting1 .........", datetime.now(), flush=True)

def collate_fn_full(examples):
    batch = {}
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    labels = torch.tensor([example["labels"] for example in examples]).to(device) # change for one-hot multilabels
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    for key in examples[0]:
        if key not in ["pixel_values", "labels"]:
            batch[key] = [ex[key] for ex in examples]
    return batch

subset_dataloader = DataLoader(train_val_ds, collate_fn=collate_fn_full, batch_size=10)
print("Starting .........", datetime.now(), flush=True)
metric_results = {}
def merge_metric_results(mr1, mr2):
    for k in mr2:
        if k in mr1:
            for k1 in mr1[k]:
                mr1[k][k1] += mr2[k][k1]
        else:
            mr1[k] = mr2[k]
    return mr1
for _, data in enumerate(subset_dataloader):
    result_v = visualize_resnet_activation_heatmaps_batch(model, data, layer=None, alpha=0.6, save_fig=False, class_labels=class_labels.names)
    metric_results = merge_metric_results(metric_results, result_v)

all_class_ranked = sorted(metric_results.keys())
metrics = metric_results[all_class_ranked[0]].keys()
print('  \t' + ",".join(all_class_ranked) + ',mean')

for k in metrics:
    v_all = []
    s = ""
    for label_ in all_class_ranked:
        v = metric_results[label_][k]
        v_all += v
        s += "%.4f," % np.mean(v)
    s += "%.4f" % np.mean(v_all)
    print(k, '\t' + s)

# print("Avg hit, iou, and dice are %.4f, %.4f, and %.4f" % (np.mean(hits), np.mean(ious), np.mean(dices)))
print("Ending .........", datetime.now(), flush=True)
exit(0)


print("------------------ Starting to analyze -----------------")
print(datetime.now())
train_val_dataloader = DataLoader(train_val_ds, collate_fn=collate_fn, batch_size=512)

# get all the features before the last layer (fc) of the model.
v_emb = torch.tensor([], dtype=torch.float32).to(device)
v_y_true = torch.tensor([], dtype=torch.long).to(device)
v_y_pred = torch.tensor([], dtype=torch.float32).to(device)
with torch.no_grad():
    for _, vdata in enumerate(train_val_dataloader):
        vinputs, vlabels = vdata['pixel_values'], vdata['labels']
        voutputs = feature_extractor(vinputs)
        voutputs = torch.flatten(voutputs['features'], 1)
        v_emb = torch.cat((v_emb, voutputs), 0)
        v_y_true = torch.cat((v_y_true, vlabels), 0) 
        pred = model.resnet.fc(voutputs)
        v_y_pred = torch.cat((v_y_pred, pred), 0)
print(datetime.now())

def collate_fn_inter(examples):
    features_id = torch.tensor([example["features_id"] for example in examples]).to(device)
    features = torch.stack([torch.tensor(example["features"]) for example in examples]).to(device)
    preds = torch.tensor([example["preds"] for example in examples]).to(device)
    labels = torch.tensor([example["labels"] for example in examples]).to(device)
    return {"features_id": features_id, "features": features, "preds": preds, "labels": labels}

fea_labels = Dataset.from_dict({'features_id': torch.arange(v_emb.size()[0], dtype=torch.int32).to(device), 'features': v_emb.to(device), 'preds': v_y_pred.to(device), 'labels': v_y_true.to(device)}, split='train_val_features')
fea_labels_dataloader = DataLoader(fea_labels, collate_fn=collate_fn_inter, batch_size=256)
print(datetime.now())


def evaluate_feature(labels, pred, pred_masked, selected_classes = None):
    pred = torch.sigmoid(pred).cpu().numpy()
    pred_masked = torch.sigmoid(pred_masked).cpu().numpy()
    if selected_classes:
        labels = labels[:, selected_classes]
        pred = pred[:, selected_classes]
        pred_masked = pred_masked[:, selected_classes]
    roc_auc = roc_auc_score(labels, pred, average = 'micro')
    roc_auc_masked = roc_auc_score(labels, pred_masked, average = 'micro')
    return roc_auc, roc_auc_masked


# # mask the features one by one to check the performance and fairness changes
# with torch.no_grad():
#     for i in range(feature_shape):
#         masker = torch.ones(feature_shape, dtype=torch.float32).to(device)
#         masker[i] = 0.0
#         v_y_true = torch.tensor([], dtype=torch.long)
#         preds, preds_masked = torch.tensor([]), torch.tensor([])
#         for vdata in fea_labels_dataloader:
#             vfeatures, vpred, vlabels = vdata['features'], vdata['preds'].cpu(), vdata['labels'].cpu()
#             vfeatures_masked = vfeatures * masker
#             vprobs_masked = model.resnet.fc(vfeatures_masked)
#             preds = torch.cat((preds, vpred), 0)
#             preds_masked = torch.cat((preds_masked, vprobs_masked.cpu()))
#             v_y_true = torch.cat((v_y_true, vlabels), 0)
#         roc_auc, roc_auc_masked = evaluate_feature(v_y_true, preds, preds_masked)
#         print(i, roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)
#         if i % 10 == 0:
#             print(datetime.now(), flush=True)


# # # # check the distribution of features and values, whether guassian or biomodal
# print("----------------------------- Check the distribution of each feature ----------------------------")
# print(v_emb.size()) # [1k, 2048]
# v_emb_T = np.transpose(v_emb.cpu().numpy()) # [1k, 2048] --> [2048, 1k]

# # # # get the guassian distribution from the embeddings, where the embeddings are following guassian distribution and applying relu.
# # mu_est = np.median(v_emb_T.flatten())
# # sigma_est = np.sqrt(np.mean((v_emb_T.flatten() - mu_est) ** 2))
# # print("Estimated mu:", mu_est) 
# # print("Estimated sigma:", sigma_est)
# # # Estimated mu: 0.44262746
# # # Estimated sigma: 0.31012407
# # exit(0)

# from scipy.stats import shapiro, normaltest, gaussian_kde
# import seaborn as sns
# import matplotlib.pyplot as plt

# feature_id = 1578
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# sns.histplot(v_emb_T[feature_id], alpha=0.25, kde=True, bins=30, ax=axes[0, 0])
# ax=axes[0, 0].set_title("original")

# replace_values = np.random.normal(loc = 0.0, scale = 1.0, size=v_emb_T[feature_id].size)
# v_emb_T_ = np.clip(v_emb_T[feature_id] + replace_values, a_min=0, a_max=10000)
# sns.histplot(v_emb_T_, alpha=0.25, kde=True, bins=30, ax=axes[0, 1])
# ax=axes[0, 1].set_title("mask by N(0,1)")

# replace_values = np.random.normal(loc = 0.0, scale = 10.0, size=v_emb_T[feature_id].size)
# v_emb_T_ = np.clip(v_emb_T[feature_id] + replace_values, a_min=0, a_max=10000)
# sns.histplot(v_emb_T_, alpha=0.25, kde=True, bins=30, ax=axes[1, 0])
# ax=axes[1, 0].set_title("mask by N(0,10)")

# replace_values = np.random.normal(loc = 0.0, scale = 100.0, size=v_emb_T[feature_id].size)
# v_emb_T_ = np.clip(v_emb_T[feature_id] + replace_values, a_min=0, a_max=10000)
# sns.histplot(v_emb_T_, alpha=0.25, kde=True, bins=30, ax=axes[1, 1])
# ax=axes[1, 1].set_title("mask by N(0,100)")

# plt.tight_layout()
# plt.savefig("generate_plot2_feature1578_dis.pdf", format='pdf', bbox_inches="tight")
# plt.show()
# exit(0)

# # Normality tests
# for i in range(v_emb_T.shape[0]):
#     data = v_emb_T[i]
#     print(i, ":")
#     print("Shapiro-Wilk Test p-value:", shapiro(data)[1])  # p < 0.05 suggests non-normality
#     print("D’Agostino and Pearson’s Test p-value:", normaltest(data)[1])

# exit(0)



# # Clustering features
# # Step 1: get features on the selected images (for now about 1k) --> get 1k * 2028 (for resnet)
# # Step 2: for each feature, the vector has 1k dimensions. Using K-means to get the feature clusters
print(datetime.now())
v_emb_T = np.transpose(v_emb.cpu().numpy()) # [1k, 2048] --> [2048, 1k]
K = 100
print("--------------- KMeans K=%d ---------------" % K)
kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(v_emb_T)
clusters = np.array(kmeans.labels_)
clusters_num = np.max(clusters) + 1
# for i in range(K):
#     print(i, "\t:", np.where(clusters == i))
# exit(0)
# print(datetime.now())

# # # check delta_auc
# # selected_classes = [class_labels.names.index('Atelectasis')] # only select Atelectasis labels
# with torch.no_grad():
#     for i in range(clusters_num):
#         masker = torch.tensor(np.where(clusters == i, 0, 1)).to(device)
#         v_y_true = torch.tensor([], dtype=torch.long)
#         preds, preds_masked = torch.tensor([]), torch.tensor([])
#         for vdata in fea_labels_dataloader:
#             vfeatures, vpred, vlabels = vdata['features'], vdata['preds'].cpu(), vdata['labels'].cpu()
#             vfeatures_masked = vfeatures * masker
#             vprobs_masked = model.resnet.fc(vfeatures_masked)
#             preds = torch.cat((preds, vpred), 0)
#             preds_masked = torch.cat((preds_masked, vprobs_masked.cpu()))
#             v_y_true = torch.cat((v_y_true, vlabels), 0)
#         # roc_auc, roc_auc_masked = evaluate_feature(v_y_true, preds, preds_masked)
#         roc_auc, roc_auc_masked = evaluate_feature(v_y_true, preds, preds_masked, selected_classes=selected_classes)
#         # print(i, np.sum(clusters == i), roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)
#         print(i, roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)
#         if i % 10 == 0:
#             print(datetime.now(), flush=True)

# # # Detect Bias Features
# # Split: Given a feature cluster (containing k features), devide the samples into two splits.
# # Step 1: Given the model (with the features), get the performance of two splits, and also the fairness of performance of two splits.
# # Step 2: Given the model (without the features), get the performance of two splits, and also the fairness of performance of two splits.
# # Analysis: 
# # (1) Step 1 only. If the performance of two splits are very different (very unfair) and one performs very good --> biased feature. Too idealy.
# # (2) Step 1 + 2. If the features affect significantly the predictions, and fairness for method w/o features are better than that for method w/ features. Make it more robust to high-variance scenario.

# Given v_emb_T # [2048, 1k]
print("***************** Using mean masker *****************")
# print("***************** Using gaussian masker *****************")
selected_classes_label = "Pneumonia"
selected_classes = [class_labels.names.index(selected_classes_label)] # only select Atelectasis labels
print("----------------- AUC results are only based on category %s ----------------" % str(selected_classes))
with torch.no_grad():
    for i in range(clusters_num):
        # split step
        feature_selected = torch.tensor(np.where(clusters == i, True, False))
        v_emb_T_selected_fea = v_emb_T[feature_selected] # [num_fea_in_cluster, 1k]
        mean_values = v_emb_T_selected_fea.mean(0) # [1k]
        mean_est = mean_values.mean()
        split = mean_values > mean_est # [1k] with True/False
        sigma_est = np.sqrt(np.mean((v_emb_T_selected_fea.flatten() - mean_est) ** 2))
        # set up for getting the performance, e.g., masker for masking the feature
        keeper = torch.tensor(np.where(clusters == i, 1, 0)).to(device)
        masker = torch.tensor(np.where(clusters == i, 0, 1)).to(device) # set masker to be 0, alternatively
        v_y_true = torch.tensor([], dtype=torch.long)
        # preds_split_1, preds_split_2 = torch.tensor([]), torch.tensor([])
        # preds_masked_split_1, preds_masked_split_2 = torch.tensor([]), torch.tensor([])
        preds, preds_masked = torch.tensor([]), torch.tensor([])
        vfeatures_ids_all = torch.tensor([])
        for vdata in fea_labels_dataloader:
            vfeatures_ids, vfeatures, vpred, vlabels = vdata['features_id'].cpu(), vdata['features'], vdata['preds'].cpu(), vdata['labels'].cpu()
            # vfeatures_masked = vfeatures * masker

            # replace_values = torch.normal(mean = mean_est, std = sigma_est, size=vfeatures.size()).to(device)
            # vfeatures_masked = torch.clamp(vfeatures * masker + replace_values * keeper, min=0.0)

            # replace_values = torch.normal(mean = 0.0, std = 10.0, size=vfeatures.size()).to(device)
            # vfeatures_masked = torch.clamp(vfeatures + replace_values * keeper, min=0.0)

            replace_values = 2 * mean_est - vfeatures # change the value to be another side of the mean
            vfeatures_masked = torch.clamp(vfeatures * masker + replace_values * keeper, min=0.0)
            
            vprobs_masked = model.resnet.fc(vfeatures_masked)
            preds = torch.cat((preds, vpred), 0)
            preds_masked = torch.cat((preds_masked, vprobs_masked.cpu()))
            v_y_true = torch.cat((v_y_true, vlabels), 0)
            vfeatures_ids_all = torch.cat((vfeatures_ids_all, vfeatures_ids), 0)
        
        split_signs = split[vfeatures_ids_all.int()]
        preds_split_1, preds_split_2 = preds[split_signs], preds[~split_signs]
        preds_masked_split_1, preds_masked_split_2 = preds_masked[split_signs], preds_masked[~split_signs]
        roc_auc_split_1, roc_auc_masked_split_1 = evaluate_feature(v_y_true[split_signs], preds_split_1, preds_masked_split_1, selected_classes=selected_classes)
        roc_auc_split_2, roc_auc_masked_split_2 = evaluate_feature(v_y_true[~split_signs], preds_split_2, preds_masked_split_2, selected_classes=selected_classes)
        roc_auc, roc_auc_masked = evaluate_feature(v_y_true, preds, preds_masked, selected_classes=selected_classes)
        # print(i, np.sum(clusters == i), roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)
        print(i, roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)
        print("\tsplit_1", split_signs.sum(), roc_auc_split_1, roc_auc_masked_split_1, roc_auc_split_1 - roc_auc_masked_split_1)
        print("\tsplit_2", len(split_signs) - split_signs.sum(), roc_auc_split_2, roc_auc_masked_split_2, roc_auc_split_2 - roc_auc_masked_split_2)
        print("Step 1:", roc_auc_split_1 - roc_auc_split_2)
        print("Step 2:", roc_auc_masked_split_1 - roc_auc_masked_split_2)
        if i % 10 == 0:
            print(datetime.now(), flush=True)




# # # only masking one feature does not change much on AUC, trying the top-k combination
# # top 50 features
# with torch.no_grad():
#     selected_fea_indices = [1180, 1404, 528, 1451, 1172, 1816, 1266, 2034, 1663, 666]
#     print("selected top unimportant %d features: %s" % (len(selected_fea_indices), str(selected_fea_indices)))
#     masker = torch.ones(feature_shape, dtype=torch.float32).to(device)
#     masker[selected_fea_indices] = 0.0
#     v_y_true = torch.tensor([], dtype=torch.long)
#     preds, preds_masked = torch.tensor([]), torch.tensor([])
#     for vdata in fea_labels_dataloader:
#         vfeatures, vpred, vlabels = vdata['features'], vdata['preds'].cpu(), vdata['labels'].cpu()
#         vfeatures_masked = vfeatures * masker
#         vprobs_masked = model.resnet.fc(vfeatures_masked)
#         preds = torch.cat((preds, vpred), 0)
#         preds_masked = torch.cat((preds_masked, vprobs_masked.cpu()))
#         v_y_true = torch.cat((v_y_true, vlabels), 0)
#     roc_auc, roc_auc_masked = evaluate_feature(v_y_true, preds, preds_masked)
#     print(roc_auc, roc_auc_masked, roc_auc - roc_auc_masked)

# print(datetime.now())


# importances = []
# for i in range(feature_shape): # there are in total 512 features
#     masker = torch.ones(feature_shape, dtype=torch.float32).to(device)
#     masker[i] = 0.0
#     with torch.no_grad():
#         v_y_true = torch.tensor([], dtype=torch.long)
#         v_y_pred = torch.tensor([])
#         for _, vdata in enumerate(train_val_dataloader):
#             vinputs, vlabels = vdata['pixel_values'], vdata['labels'].cpu()
#             print('h1', vinputs.shape)
#             voutputs = feature_extractor(vinputs)
#             print('h2', voutputs['features'].shape)
#             vfeatures_maked = torch.flatten(voutputs['features'], 1) * masker
#             vprobs = model.resnet.fc(vfeatures_maked)
#             v_y_pred = torch.cat((v_y_pred, vprobs.cpu()), 0)
#             v_y_true = torch.cat((v_y_true, vlabels), 0)
#         masked_preformance = multi_label_metrics(v_y_pred, v_y_true.numpy(), verbose=0)
#         # importances.append(all_feature_performance['f1'] - masked_preformance['f1'])
#     if i >= 1:
#         break

# # check if the extracted features are correct? vprobs == vprobs_final
# with torch.no_grad():
#     v_y_true = torch.tensor([], dtype=torch.long)
#     v_y_pred = torch.tensor([])
#     for _, vdata in enumerate(train_val_dataloader):
#         vinputs, vlabels = vdata['pixel_values'], vdata['labels'].cpu()
#         voutputs = feature_extractor(vinputs)
#         voutputs = torch.flatten(voutputs['features'], 1)
#         vprobs = model.resnet.fc(voutputs)
#         vprobs_final = model(vinputs)
#         print(vprobs, vprobs_final)