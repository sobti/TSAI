from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelTrainer import ModelTrainer
import Resnet as rn

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]
      
    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if relu:
        l.append(nn.ReLU())
      if dropout>0:
        l.append(nn.Dropout(dropout))  
      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)
                 
    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, dataloader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0, LossType='CrossEntropyLoss', tb=None):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, dataloader, statspath, scheduler, batch_scheduler, L1lambda, LossType, tb)
      self.trainer.run(epochs)

    def resumerun(self, epochs):
        self.trainer.run(epochs)
        
    def modelload(self, path):
        self.load_state_dict(torch.load(path))
        
    def stats(self):
      return self.trainer.stats if self.trainer else None

class Cfar10Net(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 32, dropout=dropout_value)
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = self.create_conv2d(32, 64, padding=2, dilation=2, dropout=dropout_value)
        self.conv4 = self.create_conv2d(64,64, groups=64, dropout=dropout_value)
        self.conv5 = self.create_conv2d(64,128, dropout=dropout_value)
        self.conv6 = self.create_conv2d(128,128, dropout=dropout_value)
        self.conv7 = self.create_conv2d(128, 256, dropout=dropout_value)
        self.dconv1 = self.create_conv2d(16, 32, dilation=2, padding=2)
        self.conv8 = self.create_conv2d(256, 10, kernel_size=(1,1),padding=0, bn=False, relu=False)
        self.gap = nn.AvgPool2d(kernel_size=(3,3))
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool1(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class Resnet_Architecture(Net):
    def __init__(self, name="Model", Model_Layers=18, num_classes=10):
        super(Resnet_Architecture, self).__init__(name)
        self.Model_Layers=Model_Layers
        self.num_classes=num_classes
        
        if self.Model_Layers==18:
            self.resnet_arch = rn.ResNet(rn.BasicBlock, [2,2,2,2],self.num_classes)
        elif self.Model_Layers==34:
            self.resnet_arch = rn.ResNet(rn.BasicBlock, [3,4,6,3],self.num_classes)
        elif self.Model_Layers==50:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,4,6,3],self.num_classes)
        elif self.Model_Layers==101:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,4,23,3],self.num_classes)
        elif self.Model_Layers==152:
            self.resnet_arch = rn.ResNet(rn.Bottleneck, [3,8,36,3],self.num_classes)
        else:
            raise ValueError('Please choose the value from 18,34,50,101,152.')
    
    def forward(self,x):
        return self.resnet_arch(x)
        
    def test(self):
        net = rn.ResNet18()
        y = net(torch.randn(1,3,32,32))
        print(y.size())

    # test()
    
class Resnet_Custom_Architecture(Net):
    def __init__(self, name="Model", num_classes=10):
        super(Resnet_Custom_Architecture, self).__init__(name)
        
        self.layer0=self.create_conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.layer1=self.ResBlock(in_channels=64, out_channels=128, kernel_size=(3,3),padding=1, bn=True, dropout=0, relu=True)
        self.Resb1=self.ResBlock(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bn=True, dropout=0, relu=True, rep=2)
        self.layer2=self.ResBlock(in_channels=128, out_channels=256, kernel_size=(3,3),padding=1,bn=True, dropout=0, relu=True)
        self.layer3=self.ResBlock(in_channels=256, out_channels=512, kernel_size=(3,3),padding=1,bn=True, dropout=0, relu=True)
        self.Resb2=self.ResBlock(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bn=True, dropout=0, relu=True, rep=2)
        self.pool=nn.MaxPool2d(4)
        self.linear = nn.Linear(512, num_classes, bias=False)
        
    def ResBlock(self, in_channels, out_channels, kernel_size, padding=1, bn=False, dropout=0, relu=False, rep=0):
        layer=[]
        if rep==0:
            layer.append(self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)+[nn.MaxPool2d(2)],out_channels, bn=bn, dropout=dropout, relu=relu))
        for i in range(rep):
            layer.append(self.create_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,bn=bn, dropout=dropout, relu=relu))
        return nn.Sequential(*layer)
        
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out= self.Resb1(out)+out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.Resb2(out)+out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), -1)
        return F.log_softmax(out, dim=-1)
        

class Unet(Net):
    def __init__(self, name="Model", dropout_value=0.02):
        super(Unet, self).__init__(name)

        self.convblockin = self.create_conv2d(in_channels=6, out_channels=3, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock1 = self.create_conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock2 = self.create_conv2d(in_channels=35, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock3 = self.create_conv2d(in_channels=99, out_channels=128, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock4 = self.create_conv2d(in_channels=227, out_channels=256, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock5 = nn.Sequential(
            self.create_conv2d(in_channels=256, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            self.create_conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            self.create_conv2d(32, 3, kernel_size=(1,1),padding=0, bn=False, relu=False)
        )
        self.convunet1 = self.create_conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convunet2 = self.create_conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convunet3 = self.create_conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.centerlayer = nn.Sequential(self.create_conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1,bn=True, dropout=dropout_value, relu=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(128))

        self.upsmaple1 = nn.Sequential(self.create_conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(64))
        
        self.upsmaple2 = nn.Sequential(self.create_conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(32))

        self.finalconv = nn.Sequential(self.create_conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3,3),bias=False,padding=1))

####################################Forward pass##########################################
    def forward(self,xin, yin=None):
        if isinstance(xin,dict):
            x1=xin['fgbg']
            y1=xin['bg']
        if yin!=None:
            x1=xin
            y1=yin
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
        convu4 = self.pool1(convu3)
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
        if yin!=None:
            return conv5, final
        return {'fgbgmask':conv5,'dense':final}
        
class UnetExp(Net):
    def __init__(self, name="Model", dropout_value=0.02):
        super(UnetExp, self).__init__(name)

        self.convblockin = self.create_conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock1 = self.create_conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock2 = self.create_conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock3 = self.create_conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock4 = self.create_conv2d(in_channels=224, out_channels=256, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convblock5 = nn.Sequential(
            self.create_conv2d(in_channels=256, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True),
            self.create_conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        )
        self.convblock6 = self.create_conv2d(32, 1, kernel_size=(1,1),padding=0, bn=False, relu=False)
        self.convunet1 = self.create_conv2d(in_channels=256, out_channels=192, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convunet2 = self.create_conv2d(in_channels=192, out_channels=128, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.convunet3 = self.create_conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=1,bn=True, dropout=0, relu=True)
        self.centerlayer = nn.Sequential(self.create_conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1,bn=True, dropout=dropout_value, relu=True),
            nn.ConvTranspose2d(128, 128, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(128))

        self.upsmaple1 = nn.Sequential(self.create_conv2d(in_channels=192, out_channels=156, kernel_size=(3,3), padding=1,bn=True, dropout=dropout_value, relu=True),
            nn.ConvTranspose2d(156, 128, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(128))
        
        self.upsmaple2 = nn.Sequential(self.create_conv2d(in_channels=256, out_channels=224, kernel_size=(3,3), padding=1,bn=True, dropout=dropout_value, relu=True),
            nn.ConvTranspose2d(224, 64, 2, stride=2), nn.ReLU(), nn.BatchNorm2d(64))

        self.finalconv = nn.Sequential(self.create_conv2d(in_channels=320, out_channels=256, kernel_size=(3,3), padding=1,bn=True, dropout=dropout_value, relu=True),
            self.create_conv2d(256, 1, kernel_size=(1,1),padding=0, bn=False, relu=False))

####################################Forward pass##########################################
    def forward(self,xin, yin=None):
        if isinstance(xin,dict):
            x1=xin['fgbg']
            y1=xin['bg']
        if yin!=None:
            x1=xin
            y1=yin
			
        bg_conv=self.convblockin(y1) #32
        fgbg_conv=self.convblockin(x1) #32
        cat0=torch.cat([fgbg_conv,bg_conv],1) #64
        conv1 = self.convblock1(cat0) #64
        cat1=torch.cat((conv1,fgbg_conv),1) #96 
        conv2 = self.convblock2(cat1) #96
        cat1=torch.cat((conv1,conv2,fgbg_conv),1) #192        
        conv3=self.convblock3(cat1) #256
        mask_conv=self.convblock5(conv3) #32
        mask_conv=self.convblock6(mask_conv) #1
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cat2=torch.cat((conv1,conv2,cat0),1) #224
        conv4=self.convblock4(cat2) #256
        convu1=  self.convunet1(conv4) #192
        convu2=  self.pool1(convu1) #hf
        convu2 = self.convunet2(convu2) #128
        convu3 = self.pool1(convu2) #hff
        convu3 = self.convunet3(convu3) #64
        convu4 = self.pool1(convu3) #hfff
        cent =   self.centerlayer(convu4) #128 #hff
        cat1=torch.cat([cent, convu3],1) #192
        #cat1=torch.cat([cent, F.upsample_bilinear(convu3, cent.size()[2:])], 1)
        upsam1=self.upsmaple1(cat1) #128 #hf
        cat2=torch.cat([upsam1, convu2], 1) #256
		#cat2=torch.cat([upsam1, F.upsample_bilinear(convu2, upsam1.size()[2:])], 1)
        upsam2=self.upsmaple2(cat2) #64 #orig
        cat3=torch.cat([upsam2, conv4], 1) #320
        final=self.finalconv(cat3)
        if yin!=None:
            return mask_conv, final
        return {'fgbgmask':mask_conv,'dense':final}