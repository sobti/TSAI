import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
class PyTorchImageDataset(Dataset):
    def __init__(self, image, augmentations = None):
        super().__init__()

        self.image = image
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image) # return 1 as we have only one image

    def __getitem__(self, idx):
      
       images,labels=self.image[idx]
       data = {"image":np.array(images,dtype=float)}
       if self.augmentations is not None:
          image = self.augmentations(**data)['image']
          image=torch.tensor(image)
          image=np.transpose(image,(2,1,0))
          return image,labels