import numpy as np
import albumentations as A
from albumentations import pytorch as pyt
import inspect
class Albumentations:
  def __init__(self):
    """More transforms can be find out on the link "https://albumentations.readthedocs.io/en/latest/api/augmentations.html". It contains all the augmentations
    provided by the Albumentations library. Please install the latest package of Albumentations, the colab inbuilt albumentations library does not contain pytorch features.
    There are many ways to incoporate this library, by ammending the dataset library (torch.utils.data.DataSet) library. The function name is def __getitem__. I am implementing
    with def__call__ option with transform option of torch vision by override with albumentations. The default list of transformation are ("Blur", "VerticalFlip", "HorizontalFlip",
    "Flip", "Normalize", "Transpose", "RandomCrop", "RandomGamma", "RandomRotate90", "Rotate", "ShiftScaleRotate", "CenterCrop", "OpticalDistortion", "GridDistortion", 
    "ElasticTransform", "RandomGridShuffle", "HueSaturationValue", "PadIfNeeded", "RGBShift", "RandomBrightness", "RandomContrast", "MotionBlur", "MedianBlur", "GaussianBlur",
    "GaussNoise", "GlassBlur", "CLAHE", "ChannelShuffle", "InvertImg", "ToGray", "ToSepia", "JpegCompression", "ImageCompression", "Cutout", "CoarseDropout", "ToFloat", "FromFloat",
    "Crop", "CropNonEmptyMaskIfExists", "RandomScale", "LongestMaxSize", "SmallestMaxSize", "Resize", "RandomSizedCrop", "RandomResizedCrop", "RandomBrightnessContrast", "RandomCropNearBBox",
    "RandomSizedBBoxSafeCrop", "RandomSnow", "RandomRain", "RandomFog", "RandomSunFlare", "RandomShadow", "Lambda", "ChannelDropout", "ISONoise", "Solarize", "Equalize", "Posterize",
    "Downscale", "MultiplicativeNoise", "FancyPCA", "MaskDropout", "GridDropout"). 
    /*In colab please run the ""!pip install -U git+https://github.com/albu/albumentations > /dev/null && echo "All libraries are successfully installed!".""/*"""
  
  def transArguDetails(self, *argv): 
    """The default list of transformation are ("Blur", "VerticalFlip", "HorizontalFlip", "Flip", "Normalize", "Transpose", "RandomCrop", "RandomGamma", "RandomRotate90", "Rotate", 
    "ShiftScaleRotate", "CenterCrop", "OpticalDistortion", "GridDistortion", "ElasticTransform", "RandomGridShuffle", "HueSaturationValue", "PadIfNeeded", "RGBShift", "RandomBrightness",
    "RandomContrast", "MotionBlur", "MedianBlur", "GaussianBlur", "GaussNoise", "GlassBlur", "CLAHE", "ChannelShuffle", "InvertImg", "ToGray", "ToSepia", "JpegCompression", 
    "ImageCompression", "Cutout", "CoarseDropout", "ToFloat", "FromFloat", "Crop", "CropNonEmptyMaskIfExists", "RandomScale", "LongestMaxSize", "SmallestMaxSize", "Resize",
    "RandomSizedCrop", "RandomResizedCrop", "RandomBrightnessContrast", "RandomCropNearBBox", "RandomSizedBBoxSafeCrop", "RandomSnow", "RandomRain", "RandomFog", "RandomSunFlare",
    "RandomShadow", "Lambda", "ChannelDropout", "ISONoise", "Solarize", "Equalize", "Posterize", "Downscale", "MultiplicativeNoise", "FancyPCA", "MaskDropout", "GridDropout")"""
    for operation in argv:
        if hasattr(A, operation):
            print(operation,':',inspect.getargspec(getattr(A, operation)),',')
        else:
            print('The syntax of '+operation+' is wrong, please use help() function.')
    return self
  
  def __call__(self,img):
    img=np.array(img)
    img=self.Transforms(image=img)['image']
    return img
  
  def transform(self, **kwargs):
    self.transforms=[]
    for operation, value in kwargs.items():
        if hasattr(A, operation):
            self.transforms.append(getattr(A, operation)(**value))
        elif hasattr(pyt, operation):
            self.transforms.append(getattr(pyt, operation)(**value))
        else:
            raise NameError('The operation is not valid. Please use help function')
    #self.transforms.append(ToTensorV2())
    self.Transforms=A.Compose(self.transforms)
    return self