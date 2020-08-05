# SPPnet Strategy in Pretrained mobilenet:
 :point_right:  Original paper :  [SPPnet Paper](https://arxiv.org/abs/1406.4729)
  
 :point_right:  Refernce code  :  [Sppnet](https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py)
 
 :point_right: SPPnet with mobilenet Refernce code :[mobile-spp](https://github.com/sobti/TSAI/blob/master/Drone%20Prediction/MobilenetV2_Model.ipynb)
 
## Sppnet:(Taken from paper)

 ![Sppnet](https://user-images.githubusercontent.com/42212648/89426205-857e6080-d757-11ea-8510-3147acea6a78.png)
 
Usually deep neural network requires fixed size input images. This is obtained mostly by transformation (resize and center crop) strategy. with transformation , there is possible chances that we loose important infromation from the images. Sppnet overcome the requiremnet to trtanform the images to arbitary size and hence neural net equip with SPP-net can run with images having varying sizes.Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods.

##  implenmenting SPP net:

- Sppnet is mostly implement before FC layer.

      model1 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
      
      class sppmob(nn.Module):
        def __init__(self):
           super(sppmob, self).__init__()
           count=1 
           for child in model1.children(): 
            if count==1:  
             self.conv1=child
             count=count+1 
             break;

           self.lin=nn.Linear(in_features=26880,out_features=4,bias=True)    

        def forward(self, x):
           x=self.conv1(x)
           spp=spatial_pyramid_pool(x,1,[int(x.size()[2]),int(x.size()[3])],[4,2,1])
           x=self.lin(spp)
           return x
           
  in_features=26880 -> fetaures out from Sppnet for each images

- It requires custom Dataloader - Collate_Fn needs to be customised to create data loader of variable size images as dataloader stacks similiar size images.

         def my_collate(batch):
           data={}
           data['total_drones'] = [(item['total_drones']) for item in batch]
           data['total_drones'] = (torch.Tensor(data['total_drones'][0])).unsqueeze(dim=0)
           data['labels'] = [item['labels'] for item in batch]
           data['labels'] = torch.LongTensor(data['labels'])
           return data
 
 - As mobilenet goes very deep such that image of 224 * 224 squeezes to 7 * 7 .BE alert in feeding smaller dimension images as maxpooling in Sppnet can cause issue.
 
 - Sometimes without having control on images can cause out of memory issue as image going can be having higher dimensions.

## yet to explore part:

- ** I have to make batch size = 1 for variable size images ( Needs to explore more on it) 

-  facing Cuda out of memory as few images are very large .Advisable to keep images in certain range.
