import torch
from torchsummary import summary
def model_gpuchk(torch):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)
  
  
def summ_print(model,input):
  summary(model,input_size=input)