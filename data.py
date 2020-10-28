import os
import torch
from PIL import Image
import glob
import random


class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, haze_root, transforms):
        self.haze_root = haze_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.haze_root, '*.bmp'))
        self.image_name_list += (glob.glob(os.path.join(self.haze_root, '*.jpg')))
        self.matching_dict = {}
        self.file_list = []
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, haze_image_name = self.file_list[item]
        ori_image = self.transforms(Image.open(ori_image_name))
        haze_image = self.transforms(Image.open(haze_image_name))
        return ori_image, haze_image

    def __len__(self):
        return len(self.file_list)
        
    def get_image_pair_list(self):
        for image in self.image_name_list:
            image = os.path.basename(image)
            key = image.split("_")[0] + ".jpg"

            if key in self.matching_dict.keys():
                self.matching_dict[key].append(image)
            else:
                self.matching_dict[key] = []
                self.matching_dict[key].append(image)

        for key in list(self.matching_dict.keys()):
            for hazy_image in self.matching_dict[key]:
                self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])

        random.shuffle(self.file_list)
