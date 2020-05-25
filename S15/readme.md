# Mask and Depth Prediction 
## Getting Started:
1. To make model up and running in your machine ,please follow below instruction :

   - Follow the data set creation approch from the following link : https://github.com/sobti/TSAI/blob/master/S14-15/Readme.md
         No of data in fgbg list: 400000
         No of data in bg list: 400000
         No of image in mask image: 400000
         No of image in depth image: 400000

   - After creating the dataset copy the Modlular code from to yout directory:
   
   - Copy the final weights to your dirctory :
   
   - Train the model as illustrated in python code :
    
   :+1: To save the time on training , It is highly recommened to load the dataset in colab 
   
 ## Model : 
 
 To predict the mask and depth on 400K images , i have used combination of two model.
 
   :+1:  Denset - https://pytorch.org/hub/pytorch_vision_densenet/
     
   :+1:  Unet - https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
     
  *Please refer my model in :* 	
   
     - I have used Dense net to Predict the mask and Unet(with Bilinear upsampling) to predict the depth. In Densent model , only one 
        dense layer is used having skip connection with each layer. I have coustomised the UNET model to reduce the capacity so as to 
        fit in the Colab CUDA.
    
   ###  Multiple Attempts : J
   *just to NOTE , These are RAW attempts .Final model must be cosidered for usage*
       
   - Try 1 : Dense net to predict maak: https://github.com/sobti/TSAI/blob/master/S15/3rd_Try_Mask_Densenet_Model.ipynb
     
           Model - Densenet 
           Image size - 64*64
           Loss - BCEwithLOGIT loss
           Optimizer - SGD 
           Model is trained on 
           
   - Try 2  : https://github.com/sobti/TSAI/blob/master/S15/4rth_Try_Mask_Densenet_Model.ipynb
                
           Model - Densenet + UNET
           Image Size - 64*64
           Loss - BCEwithLOGIT loss
           Optimizer - SGD 
           Model is trained on 
            
         
    

   
   
