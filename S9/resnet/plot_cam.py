import logging
import numpy as np
import matplotlib.pyplot as plt
class Plot:
 def fig(self,dat1,cam2,label,image_num=12):
  self.dat1=dat1
  self.cam2=cam2
  self.label=label
  self.image_num=image_num
  id=np.random.randint(12,size=63)
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  logger = logging.getLogger()
  old_level = logger.level
  logger.setLevel(100)
  fig = plt.figure(figsize=(100, 100))  # width, height in inches
  for i in range(image_num):
    sub = fig.add_subplot(1, image_num, i+1)
    sub.set_title(classes[label[id[i]]],fontsize=100)
    sub.imshow(cam2[id[i]], interpolation='nearest')
  fig = plt.figure(figsize=(100, 100))  # width, height in inches
  for i in range(image_num):
    sub = fig.add_subplot(1, image_num, i+1)
    sub.imshow(np.transpose(dat1[id[i]], (1, 2, 0)), interpolation='nearest')
  logger.setLevel(old_level)