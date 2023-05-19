import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self,
                 slices_dir,
                 masks_dir,
                 transforms):
        
        self.slices_dir = sorted(glob.glob(os.path.join(slices_dir, "*.png")))      
        self.masks_dir = sorted(glob.glob(os.path.join(masks_dir, "*.png")))      
        self.transforms = transforms

        assert len(self.slices_dir) == len(self.masks_dir)

        

    def __getitem__(self, idx):
        image = np.array(Image.open(self.slices_dir[idx]))
        mask = np.array(Image.open(self.masks_dir[idx]))
 
        mask[mask == 255.0] = 1.0

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = torch.unsqueeze(mask, 0)
            mask = mask.type(torch.uint8)

        return image, mask

    def __len__(self):
        return len(self.slices_dir)