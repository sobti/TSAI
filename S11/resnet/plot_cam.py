import logging
import numpy as np
import matplotlib.pyplot as plt
class Plot:
 def fig(self,data2,cam2,label,p,image_num=12):
  self.data2=data2
  self.cam2=cam2
  self.label=label
  self.p=p
  self.image_num=len(self.cam2)
  
  #id=np.random.randint(12,size=63)
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  logger = logging.getLogger()
  old_level = logger.level
  logger.setLevel(100)
  fig = plt.figure(figsize=(32, 32))  # width, height in inches
  for i in range(self.image_num):
    sub = fig.add_subplot(1, self.image_num, i+1)
    sub.set_title(classes[self.p[i]],fontsize=16)
    sub.imshow(self.cam2[i], interpolation='nearest')
  fig = plt.figure(figsize=(32, 32))  # width, height in inches
  for i in range(self.image_num):
    sub = fig.add_subplot(1, self.image_num, i+1)
    sub.set_title(classes[self.label[i]],fontsize=16)
    sub.imshow(np.transpose(self.data2[i], (1, 2, 0)), interpolation='nearest')
  logger.setLevel(old_level)