####  All credit goes to : https://github.com/AaronCCWong/Show-Attend-and-Tell
###### Image Captioning:
The goal of image captioning is to generate a trusted description for a given image. So we need to ensure the correctness of the objects, attribute information, semantic information, and position relationship information in the description. 
 
Subtask in Image captioning :

1. Understanding the image, acquiring the relevant information correctly with Encoder such as VGG19 . RESNET or DENSENET etc
2. Generating description based on the understanding of the image.(Attentiona and LSTM) 

We can use an encoder to automatically extract image features, and then use LSTM's as decoders to generate descriptions. Encoder-Decoder structure have become the basic framework of image captioning.

##### Show Attend and tell : https://arxiv.org/abs/1502.03044
It is an extension, which introduces a visual attention mechanism based on the Encoder-Decoder structure, which can dynamically focus on the salient regions of the image during the process of generating descriptions in Decoder.

This model also uses a CNN as Encoder to extract L vectors of K dimensions from the image, each vector corresponds to a portion of the image. But unlike the SNT model, this model uses the underlying convolutional layer output instead of the final fully connected layer output as the image feature vector:
                                                               a={a1,a2,...,aL},ai∈RK
In the decoder part, SANT also uses LSTM for description generation. But this model needs to use the image-based feature vector a for each step t to generate the context vector.
                                                               zt=∑Li=1αtiai
                                                               
