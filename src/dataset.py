from torch.utils.data import Dataset
import os
import numpy as np
from utils import *
import cv2

class PyDataset(Dataset):
    def __init__(self, indoor, indoor_gray, size, event_transform=None,
                 image_transform=None):
        self.size =
        indoor_ = os.listdir(indoor)
        indoor_.sort(key=lambda x: int(x[:-4]))
        indoor_gray_ = os.listdir(indoor_gray)
        indoor_gray_.sort(key=lambda x: int(x[:-4]))

        self.files = []
        s = ['','1','2']
        for i in range(len(self.event_ids) - 1):
            event_pre_id = [os.path.join(root, event_path + i, self.event_ids[i]) for i in s]
            event_next_id = [os.path.join(root, event_path + i, self.event_ids[i+1]) for i in s]

            image_pre_id = [os.path.join(root, grayscale_path + i, self.image_ids[i]) for i in s]
            image_next_id = [os.path.join(root, grayscale_path + i, self.image_ids[i+1]) for i in s]

            
            self.files.append({"event_pre": event_pre_id,
                               "event_next": event_next_id,
                               "image_pre": image_pre_id,
                               "image_next": image_next_id}
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        for id1, id2, in zip(data_path["event_pre"], data_path["event_next"]):
            event_pre = ArrayToTensor(cv2.resize(cv2.imread(id1), self.size[c])).permute((2, 0, 1)).float()
            event_next = ArrayToTensor(cv2.resize(cv2.imread(id2), self.size[c])).permute((2, 0, 1)).float()
            event.append(torch.cat((event_pre, event_next), 0))
            c += 1
        image_0 = []
        image_1 = []
        c = 0
        for id1, id2 in zip(data_path["image_pre"], data_path["image_next"]):
            image_pre = ArrayToTensor(cv2.resize(cv2.imread(id1), self.size[c])).permute((2, 0, 1))
            image_next = ArrayToTensor(cv2.resize(cv2.imread(id2),self.size[c])).permute((2, 0, 1))
            image_0.append([image_pre, image_next])
            image_1.append(torch.cat((image_pre, image_next), 0))
            c += 1
        return event, image_0, image_1







