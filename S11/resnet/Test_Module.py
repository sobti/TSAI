import torch.nn as nn
import torch
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, dropout=0.1):
        super(Net, self).__init__()
        self.dropout_val = dropout
        self.bias = False
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(16), nn.Dropout(self.dropout_val)) 
        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(32), nn.Dropout(self.dropout_val))
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=51, out_channels=64, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(64), nn.Dropout(self.dropout_val))
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=115, out_channels=128, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(128), nn.Dropout(self.dropout_val))
        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=243, out_channels=256, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(256), nn.Dropout(self.dropout_val))
        self.convblock7 = nn.Conv2d(in_channels=448, out_channels=64, kernel_size=(1, 1), padding=0, bias=self.bias)
        
        self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(512), nn.Dropout(self.dropout_val))
        self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(1024), nn.Dropout(self.dropout_val))
        self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=self.bias),nn.ReLU(),nn.BatchNorm2d(256), nn.Dropout(self.dropout_val))
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8)) # output_size = 1
        self.convblock11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=self.bias)
        self.linear1 = nn.Linear(1024,10)
        

    def forward(self, input):
      
      x2 = self.convblock1(input) #input - 32; Output - 32; RF - 3; output channels - 32
      x3 = self.convblock2(torch.cat([x2,input],1)) #input - 32; Output - 32; RF - 5; output channels - 32
      x4 = self.pool1(torch.cat([x3, x2, input],1)) #input - 32; Output - 16; RF - 6; output channels - 32
      x5 = self.convblock4(x4)#input - 16; Output - 16; RF - 14; output channels - 64
      
      x6 = self.convblock5(torch.cat([x5, x4],1))#input - 16; Output - 16; RF - 18; output channels - 64
      x7 = self.convblock6(torch.cat([x6, x5, x4], 1))#input - 16; Output - 16; RF - 22; output channels - 64
      x8 = self.pool1(torch.cat([x7, x6, x5], 1))#input - 16; Output - 8; RF - 24; output channels - 64
      #x = self.convblock6(x)
      #x81 = self.convblock11(x8)
      x9 = self.convblock7(x8)#input - 8; Output - 8; RF - 28; output channels - 128
      x10 = self.convblock8(torch.cat([x9, x8], 1))#input - 8; Output - 8; RF - 36; output channels - 128
      x11 = self.convblock9(torch.cat([x10, x9, x8], 1))
      #x = self.pool1(x) #input - 8; Output - 4; RF - 40; output channels - 10
      #x = self.convblock9(x) #input - 4; Output - 4; RF - 56; output channels - 10
      #x = self.convblock10(x)
      #x = self.convblock10(x)
      x12 = self.gap(x11)
      x12 = x12.view(-1, 1024)
      x13 = self.linear1(x12)
      #x14 = x13.view(-1, 10)
      return F.log_softmax(x13)
