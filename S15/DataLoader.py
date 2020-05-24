import torch

class DataLoader:
  """
  Helper class to load test and train data
  """
  def __init__(self, shuffle=True, batch_size=128, seed=1, num_workers=4, pin_memory=True ):
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    
    if cuda:
      torch.cuda.manual_seed(seed)

    # dataloader arguments - something you'll fetch these from cmdprmt
    self.dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=shuffle, batch_size=batch_size)
    self.cuda=cuda

  def load(self, data):
    return torch.utils.data.DataLoader(data, **self.dataloader_args)
  
  @property
  def device(self):
    return torch.device("cuda" if self.cuda else "cpu")