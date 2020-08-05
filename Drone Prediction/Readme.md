# SPPnet Strategy in Pretrained mobilenet:
 :point_right:  Original paper :  [SPPnet Paper](https://arxiv.org/abs/1406.4729)
  
 :point_right:  Refernce code  :  [Sppnet](https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py)
 
## Sppnet:(Taken from paper)

 ![Sppnet](https://user-images.githubusercontent.com/42212648/89426205-857e6080-d757-11ea-8510-3147acea6a78.png)
 
Usually deep neural network requires fixed size input images. This is obtained mostly by transformation (resize and center crop) strategy. with transformation , there is possible chances that we loose important infromation from the images. Sppnet overcome the requiremnet to trtanform the images to arbitary size and hence neural net equip with SPP-net can run with images having varying sizes.Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods.

##  implenmenting SPP net:

- Sppnet is mostly implement before FC layer.

- It requires custom Dataloader - Collate_Fn needs to be customised to create data loader of variable size images as dataloader stacks similiar size images.

- ** I have to make batch size = 1 for variable size images ( Needs to explore more on it) 

- facing Cuda out of memory as few images are very large .Advisable to keep images in certain range.
