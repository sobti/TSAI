from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)
        
class SchoolDataset(Dataset):
    def __init__(self, data_root, filenames, transform=None):
        self.data_root=data_root
        self.paths=filenames
        self.transform=transform

    def __getitem__(self, index):
        fgbg_mask_images=Image.open(self.data_root/'fgbgmask'/self.paths[index]).resize((224,224))
        fgbg_depth_images=Image.open(self.data_root/'dense'/self.paths[index])
        fgbg_images=Image.open(self.data_root/'fgbg'/self.paths[index]).resize((224,224))
        bg_images=Image.open(self.data_root/'bg'/(self.paths[index].parent.parent.name+'.jpg')).resize((224,224))
        
        if self.transform:
            fgbg_mask_images=self.transform['fgbg_mask'](np.asarray(fgbg_mask_images).reshape(224,224,1))
            fgbg_depth_images=self.transform['fgbg_dense'](np.asarray(fgbg_depth_images).reshape(224,224,1))
            fgbg_images=self.transform['fgbg'](fgbg_images)
            bg_images=self.transform['bg'](bg_images)
        return {'fgbg':fgbg_images, 'bg':bg_images}, {'fgbgmask':fgbg_mask_images, 'dense':fgbg_depth_images}

    def __len__(self):
        return len(self.paths)