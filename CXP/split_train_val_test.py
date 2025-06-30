import os
import random
random.seed(517)

from zipfile import ZipFile
from pathlib import Path

file_path = '/var/scratch/jhuang2/data/chexpertchestxrays-u20210408/'
# filename = "CheXpert-v1.0 batch 1 (validate & csv)"
filename = "CheXpert-v1.0 batch 2 (train 1)"
img_zip_path = {}
with ZipFile(file_path + filename + '.zip') as myzip:
    # print(myzip.namelist(), len(myzip.namelist()))
    # myzip.printdir()
    images_zip_path = [x for x in myzip.namelist() if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for x in images_zip_path:
        # print(x.strip().split('/'))
        ori_x = 'CheXpert-v1.0/train/' + '/'.join(x.strip().split('/')[1:])
        img_zip_path[ori_x] = x
        print(x, ori_x)
        if len(img_zip_path) >= 10:
            break

    # subfolders = set()
    # for f in myzip.namelist():
    #     folder = Path(f).parent
    #     if folder == Path(filename):
    #         subfolders.add(f)

    # print("\n".join([str(x) for x in sorted(subfolders)]))
    
            



# file_path = '/var/scratch/jhuang2/data/chexpertchestxrays-u20210408/'

# save_path = './split_random/'
# f = open(file_path + 'train_cheXbert.csv', 'r')
# fp1 = open(save_path + 'train_val.csv', 'w')
# fp2 = open(save_path + 'test.csv', 'w')

# info = f.readline()
# fp1.write(info)
# fp2.write(info)

# while True:
#     line = f.readline()
#     if not line:
#         break
#     if random.random() <= 0.1:
#         fp2.write(line)
#     else:
#         fp1.write(line)
        
# fp1.close()
# fp2.close()
# f.close()

