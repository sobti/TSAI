Pytorch Implemetation of Cifar - 10 <br /> 

Code: https://github.com/sobti/TSAI/blob/master/S11/resnet_model_album_cam__ocp_90_.ipynb

Custom model : https://github.com/sobti/TSAI/tree/master/S11/resnet

Dataset used - Cifar - 10<br />
Test Accuracy - 90.53%<br />
Epocs ran - 24<br />
Major Technique :<br />
 1. Image Augmentation.<br />
 2. One Cycle LR policy with Cyclic Momentum ( OCP By Leslie Smith) <br />
 Model Architecture :<br />
 
         Layer (type)                Output Shape         Param # 
         
================================================================ 

            Conv2d-1           [-1, 64, 32, 32]           1,728 
            
       BatchNorm2d-2           [-1, 64, 32, 32]             128
       
              ReLU-3           [-1, 64, 32, 32]               0 
              
            Conv2d-4          [-1, 128, 32, 32]          73,728
            
       BatchNorm2d-5          [-1, 128, 32, 32]             256 
       
              ReLU-6          [-1, 128, 32, 32]               0 
              
         MaxPool2d-7          [-1, 128, 16, 16]               0 
         
            Conv2d-8          [-1, 128, 16, 16]         147,456 
            
              ReLU-9          [-1, 128, 16, 16]               0 
              
      BatchNorm2d-10          [-1, 128, 16, 16]             256 
      
           Conv2d-11          [-1, 128, 16, 16]         147,456 
           
             ReLU-12          [-1, 128, 16, 16]               0
             
      BatchNorm2d-13          [-1, 128, 16, 16]             256 
      
           Conv2d-14          [-1, 256, 16, 16]         294,912 
           
             ReLU-15          [-1, 256, 16, 16]               0 
             
      BatchNorm2d-16          [-1, 256, 16, 16]             512 
      
        MaxPool2d-17            [-1, 256, 8, 8]               0 
        
           Conv2d-18            [-1, 512, 8, 8]       1,179,648 
           
      BatchNorm2d-19            [-1, 512, 8, 8]           1,024 
      
             ReLU-20            [-1, 512, 8, 8]               0 
             
        MaxPool2d-21            [-1, 512, 4, 4]               0 
        
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
           
             ReLU-23            [-1, 512, 4, 4]               0
             
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
      
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
           
             ReLU-26            [-1, 512, 4, 4]               0
             
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
      
        MaxPool2d-28            [-1, 512, 1, 1]               0
        
           Linear-29                   [-1, 10]           5,130
           
================================================================<br />

Total params: 6,573,130<br />

Special Thanks to Rohan Shravan 
 
