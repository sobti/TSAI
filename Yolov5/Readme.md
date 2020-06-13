## Train Cutsom data in Yolov5:

Yolo family is added one more dimension in their closet .Yolov5 is out .Thanks to [Ultralytics](https://github.com/ultralytics/yolov5) team to put all the repository in order.

### All credit goes to **Ultralytics** team

### Steps needed to train Cutsom data on volov5

#### Step 1:
- Create Custom dataset

   Use below link to create custom dataset: < Explaining how to custom dataset is not in scope of this tutorial >
 
   http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
   
#### Step 2:
 
- Preparation before running :

  1. clone the repo:
  
         https://github.com/ultralytics/yolov5  
  
  2. Download the weights from below folder:
  
         https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J
   
  3. Create a "weights" folder and put the downloaded weight.
  
  4. create one more "weights" folder just outside Yolov5 folder. ( This will save the best and last weights after training )
  
  5. Go inside yolov5 folder then to model folder and change the "nc"(Number of clasess) as per requirement .for example , i chnaged         "yolov5s.yaml" file for training one class .so nc = 1
  
  6. Go to data folder and provide the input(test/train) folder , number of clasess and class names: Example:
  
         # train and val datasets (image directory or *.txt file with image paths)
         train: ./dataset/images/
          val: ./dataset/images/

         # number of classes
         nc: 1

         # class names
         names: ['spongebob']
         
   7. Create "dataset" folder with images and labels as subfolder ( In images folder you can either put actual images or .txt file that        contains image location.  
   
   8. Run below command for training:
   
             ! python yolov5/train.py --img 640 --batch 16 --epochs 300 --data ./yolov5/data/custom128.yaml --cfg  
             ./yolov5/models/yolov5s.yaml --weights './yolov5/weights/yolov5s.pt'       
         
   8. Create a folder "inference " with images and output as subfolder( you can put images or video inside the images folder to detect).
   
   9. folder structure will be some thning like this.< you can use your own model sturcture .i prefered below one)  
   
          dataset(input folder) ---yolov5(main folder)----inference(output folder)---weights(last.pt) ->at same level
                                 -subfolder for yolov5
    
   Link : https://github.com/sobti/TSAI/blob/master/Yolov5/Yolov5_custom.ipynb
   
   **Yolov5 is very fast - When i trained 500 images with Yolov3 , it took 1.20 sec for a epoch but with Yolov5 , i am able to trained       in 22 sec per epoch .that is 300 epoc in less the 2 hrs**
   
   **Yolov5 can predict 140 FPS . For me it was 110 FPS(APPX) .AMAAAZZZINNGGGG Speed**
    

 
 
