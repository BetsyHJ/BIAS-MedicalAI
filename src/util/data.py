from datasets import load_dataset, Image, Dataset, DatasetDict, concatenate_datasets
from datasets.features import ClassLabel, Sequence
from PIL import Image as PIL_Image
import os
import numpy as np
import pandas as pd
import torch

from PIL import Image as PIL_Image

from datetime import datetime
# import multiprocessing as mp
# from multiprocessing import Pool
import multiprocess as mp

import zipfile


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
    labels = torch.tensor([example["labels"] for example in examples]).to(device) # change for one-hot multilabels
    return {"pixel_values": pixel_values, "labels": labels}

def read_dataset_from_folder(root_dir, label_list=None):

    # root_dir = './NIH-small/sample/'

    dataset = load_dataset('imagefolder', split='train', data_dir=os.path.join(root_dir, 'images'))
    # Add a filename column
    def add_filename(example):
        example['filename'] = os.path.basename(example['image'].filename)
        return example
    dataset = dataset.map(add_filename)

    dataset = dataset.cast_column("image", Image(mode="RGB"))

    # Load the metadata from the CSV file
    import pandas as pd
    metadata_file = os.path.join(root_dir, 'sample_labels.csv')
    # Load the metadata from the CSV file
    metadata_df = pd.read_csv(metadata_file)

    # Create a dictionary from the metadata for quick lookup
    metadata_dict = metadata_df.set_index('Image Index').to_dict(orient='index')

    # Add metadata to the dataset
    def add_metadata(example):
        filename = example['filename']
        if filename in metadata_dict:
            metadata = metadata_dict[filename]
            example.update(metadata)
        return example
    
    dataset = dataset.map(add_metadata)


    # Split "Finding Labels" into multiple labels
    metadata_df['Finding Labels'] = metadata_df['Finding Labels'].str.split('|')

    if label_list is None:
        # Get all unique labels
        all_labels = set(label for sublist in metadata_df['Finding Labels'] for label in sublist)
        # as no finding label affects so many images, most implementations remove "no finding" label.
        all_labels.remove('No Finding')
    else:
        all_labels = label_list

    # ### #TODO: only select some labels
    # all_labels = set(['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Pneumothorax']) 

    # Create a ClassLabel feature for each unique label
    class_labels = ClassLabel(names=list(all_labels))

    # Define the label feature as a sequence of ClassLabel
    labels_type = Sequence(class_labels)
    num_labels = len(class_labels.names)


    # # Remove unnecessary columns if needed
    # dataset = dataset.remove_columns(['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender'])

    # Create a dictionary from the metadata for quick lookup
    metadata_dict = metadata_df.set_index('Image Index').to_dict(orient='index')

    # Add metadata to the dataset, including the sequence of class labels
    def add_metadata(example):
        filename = example['filename']
        if filename in metadata_dict:
            metadata = metadata_dict[filename]
            example.update(metadata)
            # example['labels_list'] = [class_labels.str2int(label) if label in class_labels.names else 'No Finding' for label in metadata['Finding Labels']]
            example['labels'] = [float(class_labels.int2str(x) in metadata['Finding Labels']) for x in range(num_labels)]
        return example

    # Apply the metadata and features to the dataset
    dataset = dataset.map(add_metadata)
    # %%
    # # filter data with no finding label; we can also down-sample it.
    dataset_only_finding = dataset.filter(lambda example: sum(example['labels']) >= 1.0)
    print(len(dataset), len(dataset_only_finding))
    dataset = dataset_only_finding

    # %% [markdown]
    # ### data split
    # train : valid : test with ratio of 6:2:2.
    # 

    # %%
    train_val_ds = dataset.train_test_split(test_size=0.2, seed=42)
    test_ds = train_val_ds['test']

    return train_val_ds, test_ds, class_labels


def read_NIH_large(root_dir, meta_file = 'Data_Entry_2017.csv', label_list=None, test_ds_only=False, split_dir=None):
    
    if test_ds_only:
        assert label_list is not None

    # # load train_val and test (filenames) sets
    if split_dir:
        root_dir_ = root_dir
        root_dir += split_dir + '/'
    f = open(root_dir + 'train_val_list.txt')
    train_filenames = [x.strip() for x in f.readlines()]
    f.close()
    f = open(root_dir + 'test_list.txt')
    test_filenames = [x.strip() for x in f.readlines()]
    f.close()
    if split_dir:
        root_dir = root_dir_

    # Load the metadata from the CSV file
    metadata_file = os.path.join(root_dir, meta_file)
    # Load the metadata from the CSV file
    metadata_df = pd.read_csv(metadata_file)

    # # Create a dictionary from the metadata for quick lookup
    # metadata_dict = metadata_df.set_index('Image Index').to_dict(orient='index')

    images_dir = []
    dataset = None

    image_paths = []
    filenames_cor = []
    for folder in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, folder)) and ('images_' in folder):
            images_dir.append(os.path.join(root_dir, folder) + '/images/')
            for image_file in os.listdir(images_dir[-1]): # TODO: just for check
                image_path = os.path.join(images_dir[-1] , image_file)
                if image_file.lower().endswith(('png')):
                    image_paths.append(image_path)
                    filenames_cor.append(image_file)
                
    filenames_to_path = dict(zip(filenames_cor, image_paths))
    filenames_to_path = pd.Series(data=filenames_to_path)
    
    # TODO: just for check
    metadata_df = metadata_df[metadata_df['Image Index'].isin(filenames_cor)]

    metadata_df['image'] = filenames_to_path.loc[metadata_df['Image Index']].values
    
    # Split "Finding Labels" into multiple labels
    metadata_df['Finding Labels'] = metadata_df['Finding Labels'].str.split('|')

    # Get all unique labels
    if label_list is None:
        all_labels = set(label for sublist in metadata_df['Finding Labels'] for label in sublist)
        # as no finding label affects so many images, most implementations remove "no finding" label.
        all_labels.remove('No Finding')
        # ### #TODO: only select some labels
        # all_labels = set(['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Pneumothorax']) 
    else:
        all_labels = label_list
    # Create a ClassLabel feature for each unique label
    class_labels = ClassLabel(names=list(all_labels))

    # Define the label feature as a sequence of ClassLabel
    labels_type = Sequence(class_labels)
    num_labels = len(class_labels.names)

    metadata_df['labels'] = metadata_df['Finding Labels'].map(lambda example: [float(class_labels.int2str(x) in example) for x in range(num_labels)])
    
    # only get the following info
    metadata_df = metadata_df[['image', 'Image Index', 'Finding Labels', 'Patient Gender', 'Patient Age', 'labels']]

    # from df to dict
    def df_to_dict(df, keys=['image', 'Image Index', 'Finding Labels', 'Patient Gender', 'Patient Age', 'labels']):
        data_dict = {}
        for k in keys:
            data_dict[k] = list(df[k].values)
        return data_dict
    # split metadata into train and test according to train_filenames and test_filenames
    if not test_ds_only:
        train_indices = metadata_df['Image Index'].isin(train_filenames)
        train_df = metadata_df[train_indices]
        test_df = metadata_df[metadata_df['Image Index'].isin(test_filenames)]

        # train_val_ds = Dataset.from_pandas(train_df, split='train')
        # test_ds = Dataset.from_pandas(test_df, split='test')

        print("Training and test sets:", len(train_df), len(test_df))

        train_val_ds = Dataset.from_dict(df_to_dict(train_df), split='train')
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')

        train_val_ds = train_val_ds.cast_column("image", Image(mode="RGB"))
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))

        print(train_val_ds[0])
        return train_val_ds, test_ds, class_labels

    else:
        test_df = metadata_df[metadata_df['Image Index'].isin(test_filenames)]
        print("Only load test sets:", len(test_df))
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))
        return test_ds, class_labels
    


def read_CXP(root_dir, label_list=None, test_ds_only=False, split_dir=None):
    
    if test_ds_only:
        assert label_list is not None

    # # load train_val and test (filenames) sets
    assert split_dir != None # for CXP, we randomly split the data, as their original testset is human-annotated.

    train_val_df = pd.read_csv(split_dir + 'train_val.csv')
    test_df = pd.read_csv(split_dir + 'test.csv')
    print("The number of images in train_val and test set are %d and %d" % (len(train_val_df), len(test_df)))
    
    if label_list is None:
        label_list = [ 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    # Create a ClassLabel feature for each unique label
    class_labels = ClassLabel(names=label_list)
    # Define the label feature as a sequence of ClassLabel
    labels_type = Sequence(class_labels)
    num_labels = len(class_labels.names)

    label_values = train_val_df.loc[:, label_list].values
    label_values = np.where(label_values > 0, 1.0, 0.0) # positive (1), negative (0), uncertain (-1), and unmentioned (blank) classes; we set 1 to 1, others to 0
    train_val_df['labels'] = [x for x in label_values]
    
    label_values = test_df[label_list].values
    label_values = np.where(label_values > 0, 1.0, 0.0)
    test_df['labels'] = [x for x in label_values]

    train_val_df.rename(columns={'Path':'image'}, inplace=True)
    test_df.rename(columns={'Path':'image'}, inplace=True)

    # only get the following info: Path,Sex,Age
    train_val_df = train_val_df[['image', 'Sex', 'Age', 'labels']]
    test_df = test_df[['image', 'Sex', 'Age', 'labels']]

    # from df to dict
    def df_to_dict(df, keys=['image', 'Sex', 'Age', 'labels']):
        data_dict = {}
        for k in keys:
            data_dict[k] = list(df[k].values)
        return data_dict

    # rename image path to the correct local path
    train_val_df['image'] = train_val_df['image'].map(lambda x: x.replace("CheXpert-v1.0/train/", root_dir))
    test_df['image'] = test_df['image'].map(lambda x: x.replace("CheXpert-v1.0/train/", root_dir))

    # load images
    if not test_ds_only:
        train_val_ds = Dataset.from_dict(df_to_dict(train_val_df), split='train')
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')
        
        train_val_ds = train_val_ds.cast_column("image", Image(mode="RGB"))
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))

        print(train_val_ds[0])
        print(test_ds[0])
        return train_val_ds, test_ds, class_labels

    else:
        print("Only load test sets:", len(test_df))
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))
        return test_ds, class_labels



def read_CXP_original(root_dir, label_list=None, test_ds_only=False, split_dir=None):
    
    if test_ds_only:
        assert label_list is not None

    # # load train_val and test (filenames) sets
    assert split_dir != None # for CXP, we randomly split the data, as their original testset is human-annotated.

    train_df = pd.read_csv(split_dir + 'train_cheXbert.csv')
    val_df = pd.read_csv(split_dir + 'val_labels.csv')
    test_df = pd.read_csv(split_dir + 'test_labels.csv')
    print("The number of images in train, val, and test set are %d, %d, and %d" % (len(train_df), len(val_df), len(test_df)))
    
    if label_list is None:
        label_list = [ 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    # Create a ClassLabel feature for each unique label
    class_labels = ClassLabel(names=label_list)
    # Define the label feature as a sequence of ClassLabel
    labels_type = Sequence(class_labels)
    num_labels = len(class_labels.names)

    label_values = train_df.loc[:, label_list].values
    label_values = np.where(label_values > 0, 1.0, 0.0) # positive (1), negative (0), uncertain (-1), and unmentioned (blank) classes; we set 1 to 1, others to 0
    train_df['labels'] = [x for x in label_values]
    
    # check do we need this for val and test?
    label_values = val_df[label_list].values
    label_values = np.where(label_values > 0, 1.0, 0.0)
    val_df['labels'] = [x for x in label_values]

    label_values = test_df[label_list].values
    label_values = np.where(label_values > 0, 1.0, 0.0)
    test_df['labels'] = [x for x in label_values]

    train_df.rename(columns={'Path':'image'}, inplace=True)
    test_df.rename(columns={'Path':'image'}, inplace=True)
    val_df.rename(columns={'Path':'image'}, inplace=True)

    # only get the following info: Path,Sex,Age
    print(val_df.columns, test_df.columns)
    train_df = train_df[['image', 'labels']]
    val_df = val_df[['image', 'labels']]
    test_df = test_df[['image', 'labels']]

    # from df to dict
    def df_to_dict(df, keys=['image', 'labels']):
        data_dict = {}
        for k in keys:
            data_dict[k] = list(df[k].values)
        return data_dict

    # rename image path to the correct local path
    train_df['image'] = train_df['image'].map(lambda x: x.replace("CheXpert-v1.0/train/", root_dir))
    val_df['image'] = val_df['image'].map(lambda x: x.replace("CheXpert-v1.0/valid/", root_dir))
    test_df['image'] = test_df['image'].map(lambda x: x.replace("test/", root_dir))

    # load images
    if not test_ds_only:
        train_ds = Dataset.from_dict(df_to_dict(train_df), split='train')
        val_ds = Dataset.from_dict(df_to_dict(val_df), split='val')
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')

        train_ds = train_ds.cast_column("image", Image(mode="RGB"))
        val_ds = val_ds.cast_column("image", Image(mode="RGB"))
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))

        print(train_ds[0])
        print(val_ds[0])
        print(test_ds[0])
        return train_ds, val_ds, test_ds, class_labels

    else:
        print("Only load test sets:", len(test_df))
        test_ds = Dataset.from_dict(df_to_dict(test_df), split='test')
        test_ds = test_ds.cast_column("image", Image(mode="RGB"))
        return test_ds, class_labels