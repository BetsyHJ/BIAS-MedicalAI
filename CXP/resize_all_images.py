import os
from PIL import Image
from datetime import datetime

# trainset
# file_path = '/var/scratch/jhuang2/data/chexpertchestxrays-u20210408/'
# filename = "CheXpert-v1.0 batch 1 (validate & csv)"
# batch_path = "CheXpert-v1.0 batch 2 (train 1)/"
# batch_path = "CheXpert-v1.0 batch 3 (train 2)/"
# batch_path = "CheXpert-v1.0 batch 4 (train 3)/"

# val and test (original)
file_path = '/var/scratch/jhuang2/data/chexlocalize/CheXpert/'
# batch_path = 'val/'
batch_path = 'test/'

# save to folder
save_path = "./CheXpert-v1.0/"

fixed_size = 1024

def img_resize(root_dir, save_dir):
    i = 0
    print(datetime.now())
    for folder_patient in os.listdir(root_dir):
        current_dir = os.path.join(root_dir, folder_patient)
        if os.path.isdir(current_dir) and ('patient' in folder_patient):
            for folder_study in os.listdir(current_dir):
                current_study_dir = os.path.join(current_dir, folder_study)
                if os.path.isdir(current_study_dir) and ('study' in current_study_dir):
                    for img_file in os.listdir(current_study_dir):
                        assert img_file.endswith(('jpg', 'png'))
                        if img_file.endswith(('jpg', 'png')):
                            img = Image.open(os.path.join(current_study_dir, img_file))
                            w, h = img.size
                            if (w > fixed_size) or (h > fixed_size):
                                if w > h:
                                    w_new, h_new = fixed_size, int(h / (w * 1.0 / fixed_size))
                                else:
                                    w_new, h_new = int(w / (h * 1.0 / fixed_size)), fixed_size
                            new_img = img.resize((w_new, h_new))
                            save_path = current_study_dir.replace(root_dir, save_dir)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            # print(save_path)
                            new_img.save(os.path.join(save_path, img_file))
                            i += 1
                            if i % 1000 == 0:
                                print("Processing %d images, -- %.2f" % (i, i * 1.0 / 224316), flush=True)
                                print(datetime.now())

img_resize(os.path.join(file_path, batch_path), save_path)
                
    



