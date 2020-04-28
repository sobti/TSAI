S13 - YOLO model

https://github.com/sobti/TSAI/blob/master/S13/Yolo_Spongebob.ipynb

Train = train.py

Detect = Detect.py

Check customdata folder for data set up.

Weight folder - Initial Weight are kept

cfg folder in custom data - Configuration file for Spongebob detection

Youtube sponge bob detection - https://www.youtube.com/watch?v=jqSkiNHwYl8

customdata creation:
data
--customdata
--images/
--img001.jpg
--img002.jpg
--... --labels/
--img001.txt
--img002.txt
--...
custom.data #data file
custom.names #your class names
custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train

Custom data
-- classes=1
-- train=data/customdata/custom.txt
-- test=data/customdata/custom.txt
-- names=data/customdata/custom.names

1. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3).

2. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder.

3. Search for 'filters=18' (you should get entries entries). Change 255 to (4+1+(no of class))*3

4. Search for 'classes=1' and change all three entries to 'classes=(number of classs to predict)'
Weight

1. Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
