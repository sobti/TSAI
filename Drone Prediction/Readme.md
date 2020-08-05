# SPPnet Strategy in Pretrained mobilenet:
 :point_right:  Original paper :  [SPPnet Paper](https://arxiv.org/abs/1406.4729)
  
 :point_right:  Refernce code  :  [Sppnet](https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py)
 
## Sppnet:(Taken from paper)

 ![Sppnet](https://user-images.githubusercontent.com/42212648/89426205-857e6080-d757-11ea-8510-3147acea6a78.png)

Existing deep convolutional neural networks (CNNs) require a fixed-size (e.g., 224×224) input image. This requirement is “artificial” and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. In this work, we equip the networks with another pooling strategy, “spatial pyramid pooling”, to eliminate the above requirement. Thenew network structure, called SPP-net, can generate a fixed-length representation regardless of image size/scale. Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods. On the ImageNet 2012 dataset, we demonstrate that SPP-net boosts the accuracy of a variety of CNN architectures despite their different designs 
