# Mask and Depth Prediction 
## Getting Started:
1. To make model up and running in your machine ,please follow below instruction :

   - Follow the data set creation approch from the following link :
   [Dataset](https://github.com/sobti/TSAI/blob/master/S14-15/Readme.md)
        - No of data in fgbg list: 400000
        - No of data in bg list: 400000
        - No of image in mask image: 400000
        - No of image in depth image: 400000
        
   **See sample image below** 
   
   - Copy the final weights to your dirctory : **Depth_128_Final.pt**
     
     [Final Weights](https://github.com/sobti/TSAI/blob/master/moduler/Depth_128_final.pt)
   
   - Train the model as illustrated in python code :   
   
     [Final Code](https://github.com/sobti/TSAI/blob/master/S15/Final_Combined(128%2C128)_400k_Train_Model.ipynb)
     
   - Modular Code:
      
     [Module](https://github.com/sobti/TSAI/tree/master/moduler)
     
   - Depth Prediction using ADAM and SSIM loss (Best Depth Prediction) (Special Mention)

     [Depth](https://github.com/sobti/TSAI/blob/master/S15/ResUnet_Model_SSIM_ADAM.ipynb) 
    
   :+1: To save the time on training , It is highly recommened to load the dataset in colab 
   
 ## Model : 
 
 To predict the mask and depth on 400K images , I have used combination of two model.
 
   :+1:  Denset - https://pytorch.org/hub/pytorch_vision_densenet/
     
   :+1:  Unet - https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
     
  *Please refer mymodel in modular code* -[Mymodel](https://github.com/sobti/TSAI/blob/master/moduler/mymodel.py)
  
 
   
     - I have used Dense net to Predict the mask and Unet(with Bilinear upsampling) to predict the depth. In Densent model , only one 
        dense layer is used having skip connection with each layer. I have coustomised the UNET model to reduce the capacity so as to 
        fit in the Colab CUDA.
    
   ###  Multiple Attempts : 
   *just to NOTE , These are RAW attempts .Final model must be considered for usage*
       
   - Attempt 1 : Dense net to predict mask: https://github.com/sobti/TSAI/blob/master/S15/3rd_Try_Mask_Densenet_Model.ipynb
     
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
           
   - Attempt 6 - To get the appropriate weight that can be fed into model running on 400K images , i trained the model on 128*128 with 
                  dataset haaving batches of 50K images(8 sets) . Few codes are below for reference :
                  
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(3.5L)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(last%2050K)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined(64%2C64)_1st_Try_Mask_Densenet_Model.ipynb
       - https://github.com/sobti/TSAI/blob/master/S15/Combined_Step_training_(64%2C64)_1st_Try_Mask_Densenet_Model.ipynb
      
   - There are several other attempt that i tried such as with Adam optimisor and complete UNET models . Please mail me , if    intrested on those results
   
   - Modular code : [Modular Code](https://github.com/sobti/TSAI/tree/master/moduler)
   
   ### Zip Dataset - [Dataset](https://drive.google.com/open?id=1HNaIk27unwtBCykzPh5lTsIaOKKkMAGF)
   
   :point_right: Back Gound Image : bg.7z
         
   :point_right: Dense Depth Image : Densedepth.7z
         
   :point_right: Fore Ground -Back Gound Image : fgbg.7z
         
   :point_right: Mask Image : fgbgmask.7z
         
   ## Training of 400K images :
   
   -  I have divided the dataset in 8 set of 50K each so that my model can see all the images atleast for few epocs as it is quite 
      difficult to train 400K images at one go.
      
   -  The training was of incremental manner , after each iteration of 50K images , I saved the weight .And used same weight to train
       next lot.
       
   -  **After creating final weight , i fed the Pretrained weight to 400K images for three epochs.**
   
   - **This method helped me to attain a Test set-  Average loss: 0.0278 in 3 Epochs.**
   
   - Just to **NOTE** , As training is done to small gray scale images , Depth Prediction clarity is compromised here .
   
   ### Final Weight after training 50K images in a 8 sets(**This will be fed into final training for 400K images**) - 
   
   - [Final weight for 50K sets](https://drive.google.com/open?id=1HNaIk27unwtBCykzPh5lTsIaOKKkMAGF)
   
   - In the above link , Download - Depth_128_5.pt
   
   * This weight is best suited for images having building as background and object as foreground .Please refer dataset (mentioned below) *
  
  ### Final Model link : [Final Model](https://github.com/sobti/TSAI/blob/master/S15/Final_Combined(128%2C128)_400k_Train_Model.ipynb)
  
  
  ### Final Weights-Trained on 400K images  : [Final Weights](https://github.com/sobti/TSAI/blob/master/moduler/Depth_128_final.pt)
   
  ### Sample Input Image:
  
  1. [Back Ground Images](https://github.com/sobti/TSAI/blob/master/S14-15/Sample_Bg_Images.jpg)
     
   <img src="https://github.com/sobti/TSAI/blob/master/S14-15/Sample_Bg_Images.jpg">
     
  2. [Sample Dense Depth](https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Dense_Images.jpg)
  
     <img src="https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Dense_Images.jpg">
     
  3. [Sample FGBG](https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Images.jpg)
  
     <img src="https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Images.jpg">
     
  4. [Sample Mask](https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Masks_Images.jpg)
  
     <img src="https://github.com/sobti/TSAI/blob/master/S14-15/Sample_FgBg_Masks_Images.jpg">
     
   ### Final output Images :
    
   Refer code : [code](https://github.com/sobti/TSAI/blob/master/S15/Final_Combined(128%2C128)_400k_Train_Model.ipynb)
   
  1. [Final Mask](https://github.com/sobti/TSAI/blob/master/S15/Final_Train_Mask.PNG)
     
   <img src="https://github.com/sobti/TSAI/blob/master/S15/Final_Train_Mask.PNG">
   
  2. [Final Depth](https://github.com/sobti/TSAI/blob/master/S15/Final_Train_Dense.PNG)
     
   <img src="https://github.com/sobti/TSAI/blob/master/S15/Final_Train_Dense.PNG">
   
   
  
   
   
   
   
     
  
  
          
           
           
            
         
    

   
   
