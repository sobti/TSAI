import torch
from torchsummary import summary
class summ():

 def model_gpuchk():
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda" if use_cuda else "cpu")
   print(device)
   return device
  
  
 def summ_print(model,input):
  summary(model,input_size=input)