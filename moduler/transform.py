import torch
from PIL import Image
from torchvision import datasets, transforms
class transform():


 def trans():
  
  fgbg_transform = transforms.Compose([
                                       transforms.Resize((128, 128)),
                                       transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-10,10),fill=(0)),
                                       #transforms.Grayscale(),
                                       transforms.ToTensor()
                                      
                                       #transforms.Normalize((0.49641467,), (0.2798614 ,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

  bg_transform = transforms.Compose([
                                       transforms.Resize((128, 128)),
                                       transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-10,10),fill=(0)),
                                       #transforms.Grayscale(),
                                       transforms.ToTensor()
                                       
                                       #transforms.Normalize((0.49836101,), (0.27918635,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])


  mask_transform = transforms.Compose([
                                       transforms.Resize((128, 128)),
                                       transforms.Grayscale(num_output_channels=3),
                                       #transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
                                       #transforms.RandomRotation((-10,10),fill=(0,)),
                                       transforms.ToTensor()
                                       
                                       #transforms.Normalize((0.05250582,), (0.22033702,))
                                       ])
  depth_transform = transforms.Compose([
                                       transforms.Resize((128, 128)),
                                       transforms.Grayscale(num_output_channels=3),
                                       #transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
                                       #transforms.RandomRotation((-10,10),fill=(0,)),
                                       transforms.ToTensor()
                                       
                                       #transforms.Normalize((0.05250582,), (0.22033702,))
                                       ])
  return fgbg_transform,bg_transform,mask_transform,depth_transform
  
  
  
class Myclass(torch.utils.data.Dataset):
    def __init__ (self,bg_image_dataset,fgbg_image_dataset,mask_image_dataset,depth_image_dataset,transform=None):
       #super(depthmask, self).__init__()
       self.bg_image_dataset = bg_image_dataset
       
       self.fgbg_image_dataset = fgbg_image_dataset
       self.mask_image_dataset = mask_image_dataset
       self.depth_image_dataset = depth_image_dataset
       self.transform = transform
       

    def __getitem__ (self, idx): 
         bg = Image.open(self.bg_image_dataset[idx])
         fgbg = Image.open(self.fgbg_image_dataset[idx])
         mask = Image.open(self.mask_image_dataset[idx])
         depth = Image.open(self.depth_image_dataset[idx])
         if self.transform:
           bg = self.transform[1](bg)
           fgbg = self.transform[0](fgbg)
           mask = self.transform[2](mask)
           depth = self.transform[3](depth)
           sample = {'bg' : bg, 'fgbg': fgbg, 'mask': mask, 'depth': depth}
         return sample
    
    def __len__ (self):
       return len(self.fgbg_image_dataset)