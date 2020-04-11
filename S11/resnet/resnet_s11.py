import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
 #PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]   <<<<<<<<<<<<<<<<<<<<PREP layer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
     
#X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]<<<<<<<<<<<<<<<<<<<<<Layer1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
    
#R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )
        
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<layer2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
         )
        self.pool2 = nn.MaxPool2d(2, 2)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Layer3>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
    
#R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            
        )

        self.pool4 = nn.MaxPool2d(4, 4) # output_size = 1
        self.fc1 = nn.Linear(512, 10)
       

    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        R1 = self.convblock3(x)
        
        x =x + R1
        
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.pool3(x)
        R2 = self.convblock6(x)


        x = x + R2
       
        x = self.pool4(x) 
        x = x.view(x.size(0), -1)
        x=self.fc1(x)  
        return x
      

