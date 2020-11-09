###  All credit goes to : https://github.com/AaronCCWong/Show-Attend-and-Tell
### Image Captioning:
The goal of image captioning is to generate a trusted description for a given image. So we need to ensure the correctness of the objects, attribute information, semantic information, and position relationship information in the description. 
 
Subtask in Image captioning :

1. Understanding the image, acquiring the relevant information correctly with Encoder such as VGG19 . RESNET or DENSENET etc
2. Generating description based on the understanding of the image.(Attentiona and LSTM) 

We can use an encoder to automatically extract image features, and then use LSTM's as decoders to generate descriptions. Encoder-Decoder structure have become the basic framework of image captioning.

### Show Attend and tell : https://arxiv.org/abs/1502.03044
It is an extension, which introduces a visual attention mechanism based on the Encoder-Decoder structure, which can dynamically focus on the salient regions of the image during the process of generating descriptions in Decoder.

### Architecture :

Encoder-Decoder architecture- Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

 -   Encoder - Encoder Architecture is used to obtain the feature of image provided . The features added with embedding is fed to Decoder. We have used VGG19
 -   Decoder - This is usually  LSTM cells used to generate the description.

Attention-  Attention models, or attention mechanisms, are input processing techniques for neural networks that allows the network to focus on specific aspects of a complex input, one at a time until the entire dataset is categorized.

### Data Split : https://cs.stanford.edu/people/karpathy/deepimagesent/
 Data Split can be used from above link.
 
                                                               
