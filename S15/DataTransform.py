from torchvision import transforms

class Transforms():
  """
  Helper class to create test and train transforms
  """
  def __init__(self, normalize=False, mean=None, stdev=None):
    if normalize and (not mean or not stdev):
      raise ValueError('mean and stdev both are required for normalize transform')
  
    self.normalize=normalize
    self.mean = mean
    self.stdev = stdev

  def test_transforms(self):
    transforms_list = [transforms.ToTensor()]
    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    return transforms.Compose(transforms_list)

  def train_transforms(self, pre_transforms=None, post_transforms=None):
    if pre_transforms:
      transforms_list = pre_transforms
    else:
      transforms_list = []
    transforms_list.append(transforms.ToTensor())

    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    if post_transforms:
      transforms_list.extend(post_transforms)
    return transforms.Compose(transforms_list)