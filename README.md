# TSAI <for all module , a special guidence was provided by "The School of AI" , specially thanks to Rohan Shravan>
All the model are kept in : https://github.com/sobti/TSAI/tree/master/S11/resnet
# S13 - YOLO model 
https://github.com/sobti/TSAI/blob/master/S13/Yolo_Spongebob.ipynb </br>
Train = train.py <br/>
Detect = Detect.py <br/>
Check customdata folder for data set up. <br/>
Weight folder - Initial Weight are kept <br/> 
cfg folder in custom data - Configuration file for Spongebob detection  <br/>
Youtube sponge bob detection - https://www.youtube.com/watch?v=jqSkiNHwYl8 <br/>
customdata creation:<br/>
 data <br/>
  --customdata<br/>
    --images/<br/>
      --img001.jpg<br/>
      --img002.jpg<br/>
      --...
    --labels/<br/>
      --img001.txt<br/>
      --img002.txt<br/>
      --...<br/>
    custom.data #data file<br/>
    custom.names #your class names<br/>
    custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train<br/>
    Custom data <br />
      classes=1<br />
      train=data/customdata/custom.txt<br />
      test=data/customdata/custom.txt <br />
      names=data/customdata/custom.names<br />
      For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). <br />
      Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder.<br />
      Search for 'filters=18' (you should get entries entries). Change 255 to  (4+1+(no of class))*3<br />
      Search for 'classes=1' and change all three entries to 'classes=<number of classs to predict>'<br />
      Weight<br />
      Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
