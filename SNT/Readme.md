####  All credit goes to : https://github.com/AaronCCWong/Show-Attend-and-Tell
### Image Captioning:
The goal of image captioning is to generate a trusted description for a given image. So we need to ensure the correctness of the objects, attribute information, semantic information, and position relationship information in the description. 
 
Subtask in Image captioning :

1. Understanding the image, acquiring the relevant information correctly with Encoder such as VGG19 . RESNET or DENSENET etc
2. Generating description based on the understanding of the image.(Attentiona and LSTM) 

We can use an encoder to automatically extract image features, and then use LSTM's as decoders to generate descriptions. Encoder-Decoder structure have become the basic framework of image captioning.
