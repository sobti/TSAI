import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
class GradCm:
 def img(self,data,label,net):
  self.net=net
  self.data=data
  self.label=label
  criterion=nn.CrossEntropyLoss()
  v={}
  vbck={}
  train_losses=[]
  v.clear()
  vbck.clear()
  def hook_fn(m,i,o):
   v['k']=o
   
  net.layer3.register_forward_hook(hook_fn)
  net.layer3.register_backward_hook(hook_fn)
  data=data.to('cuda')
  data.unsqueeze_(0)
  out=net(data)
  
  #Forward Hook
  
  vp=v['k'][0].squeeze(0)
  vp=vp.to('cpu')
 
 
  label=label.to('cuda')
  err = criterion(out, label)
  err.backward()
  
  #BAck hook
  
  vb=v['k'][0].squeeze(0)
  vb=vb.to('cpu')
  
  #Average Pooling
  
  grad_b = F.adaptive_avg_pool2d(vb,1)
  grad_b=grad_b.to('cpu')
  gcam_b=vp.squeeze(0)
  #Multiplying and summing features and Backprop variable
  
  gcam_b = (gcam_b*grad_b).sum(dim=0, keepdim=True)
  
  # Relu 
  gcam_b = F.relu(gcam_b)
  
  #Preparing GradCAm 
  
  gcam_b=gcam_b.unsqueeze(0)
  gcam = F.interpolate(gcam_b , [32,32], mode="bilinear", align_corners=False)
  gcam=gcam.squeeze(0)
  gcam=gcam.squeeze(0)
  gcam=gcam.detach().numpy()
  cam = np.maximum(gcam, 0)
  heatmap = cam / np.max(cam) # normalize
  cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # balck-and-white to color
  data=data.squeeze(0)
  data=data.to('cpu')
  data=data.detach().numpy()
  data=data.transpose(2,1,0)
  cam = np.float32(cam) + np.float32(data) # overlay heatmap onto the image
  cam = 255 * cam / np.max(cam)
  camy=np.uint8(cam)
  
  return camy
