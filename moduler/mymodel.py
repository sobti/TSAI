import torch.nn as nn
import torch.nn.functional as F
import torch

dropout_value=0.02
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.convblockin = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=3,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
    
        )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
    
        )
       

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=35,out_channels=64,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
    
        )
      

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=99,out_channels=128,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
    
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=227,out_channels=256,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
            
         )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=64,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1),bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32,out_channels=3,kernel_size=(1,1),bias=False),
    
        )
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.convunet1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
    
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convunet2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
    
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.convunet3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
    
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.centerlayer = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        #concat here ---torch.cat([centerlayer, F.upsample_bilinear(convblock4, centerlayer.size()[2:])], 1)) 

        
         
        self.upsmaple1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=(3,3),bias=False,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )
        
         #concat here---torch.cat([upsample2, F.upsample_bilinear(convblock2, upsample2.size()[2:])], 1))
        
        self.upsmaple2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=(3,3),bias=False,padding=1),
          
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32))
          
         # concat -- here --torch.cat([upsample2, F.upsample_bilinear(convblock2, upsample2.size()[2:])], 1))

        self.finalconv = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32, kernel_size=(3,3),bias=False,padding=1),
         
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3,3),bias=False,padding=1)
        )
#----------------------------------------------------UNET MODEL OF DEPTH PREDICTION---------------------------------------------------------------------------------------------------              
        

####################################Forward pass##########################################


    def forward(self, x1,y1):

        #x=torch.cat((x1,y1),1)

        #x= self.convblockin(x)
         
         conv1=self.convblock1(x1)

         cat0=torch.cat((conv1,x1),1)

         conv2 = self.convblock2(cat0)

         cat1=torch.cat((conv1,conv2,x1),1)
        
         conv3=self.convblock3(cat1)
        
         cat2=torch.cat((conv1,conv2,conv3,x1),1)
 
         conv4=self.convblock4(cat2)

         conv5=self.convblock5(conv4)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
         x_dep=torch.cat((conv5,y1),1)

         x= self.convblockin(x_dep)
        
        #conv11=self.convblock1(x_dep)
         convu1=  self.convunet1(x)
         convu2=  self.pool1(convu1)
         convu2 = self.convunet2(convu2)
         convu3 = self.pool1(convu2)
         convu3 = self.convunet3(convu3)
         convu4 = self.pool3(convu3)
         cent =   self.centerlayer(convu4)
        #--------------------------------------------------------------------------------------------------------------------
        #mask=self.convcent(cent)
        #---------------------------------------------------------------------------------------------------------------------
         cat1=torch.cat([cent, F.upsample_bilinear(convu3, cent.size()[2:])], 1)
         upsam1=self.upsmaple1(cat1)
         cat2=torch.cat([upsam1, F.upsample_bilinear(convu2, upsam1.size()[2:])], 1)
         upsam2=self.upsmaple2(cat2)
         cat3=torch.cat([upsam2, F.upsample_bilinear(convu1, upsam2.size()[2:])], 1)
         final=self.finalconv(cat3)
         
         return conv5,final