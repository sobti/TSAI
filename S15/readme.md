# Mask and Depth Prediction 
## Getting Started:
1. To make model up and running in your machine ,please follow below instruction :

   - Follow the data set creation approch from the following link : https://github.com/sobti/TSAI/blob/master/S14-15/Readme.md
        - No of data in fgbg list: 400000
        - No of data in bg list: 400000
        - No of image in mask image: 400000
        - No of image in depth image: 400000

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
       
   - Attempt 1 : Dense net to predict maak: https://github.com/sobti/TSAI/blob/master/S15/3rd_Try_Mask_Densenet_Model.ipynb
     
           Model - Densenet 
           Loss - BCEwithLOGIT loss
           Optimizer - SGD 
           
           
   - Attempt 2  : https://github.com/sobti/TSAI/blob/master/S15/4rth_Try_Mask_Densenet_Model.ipynb
                
           Model - Densenet + UNET
           Loss - BCEwithLOGIT loss
           Optimizer - SGD 
           
           
   - Attempt 3  : https://github.com/sobti/TSAI/blob/master/S15/MSE_5th_Try_Depth_Densenet_Model.ipynb
   
           Model - Densenet + UNET
           Loss - MSE loss
           Optimizer - SGD
           
   - Attempt 4  : https://github.com/sobti/TSAI/blob/master/S15/MSE_5th_Try_Depth_Densenet_Model.ipynb
   
           Model - Densenet + UNET
           Loss - MSE loss
           Optimizer - SGD
           
   - Attempt 5 : https://github.com/sobti/TSAI/blob/master/S15/RMSE_Combined_1st_Try_Mask_Densenet_Model.ipynb
   
           Model - Densenet + UNET
           Loss - RMSE loss
           Optimizer - RMSProp
           
   - Attempt 6 - To get the appropriate weight that can be fed into 400L images , i gave trained the model on 128*128 with 
                  dataset batch of 50K images . Few codes are below for reference :
                  
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(3.5L)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(last%2050K)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(64%2C64)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined_Step_training_(64%2C64)_1st_Try_Mask_Densenet_Model.ipynb
      
   - There are several other attempt that i tried such as with Adam optimisor and complete UNET models . Please mail me , if you are     intrested on those results
   
   - Modular code : https://drive.google.com/open?id=1HNaIk27unwtBCykzPh5lTsIaOKKkMAGF
           
   1. Final Weight - In the above link , Download - Depth_128_5.pt
   
   2. Zip Dataset -
   
         :point_right: Back Gound Image : bg.7z
         :point_right: Back Gound Image : Densedepth.7z
         :point_right: Back Gound Image : fgbg.7z
         :point_right: Back Gound Image : fgbgmask.7z
          
           
           
            
         
    

   
   
