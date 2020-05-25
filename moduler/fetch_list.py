import os
from PIL import Image
class getlist():
    
 def list():
  bg_image_list_final=[]
  fgbg_image_list=[]
  bg_image_list=[]
  mask_image_list=[]
  depth_image_list=[]
  count=1  # to restrict the data to few thousand


# Getting the BG list and replicate each background to 4000 


  for root, dirs, files in os.walk("/content/data/bg/"):
   for name in files:
      bg_image_list.append(root+name)
      bg_image_list_final=(((bg_image_list)*4000)+bg_image_list_final)
      fgbg_folder=(name.split('.'))[0]
      bg_image_list=[]
      
      

# Getting the fgbg list into Variable for data loader.

      path="/content/data/fgbg/" + fgbg_folder
  
      for root_fgbg, dirs_fgbg, files_fgbg in os.walk(path):
          for name_fgbg in files_fgbg:
             fgbg_image_list.append(root_fgbg +'/'+ name_fgbg)

# Getting the mask list into Variable for data loader.

      path="/content/data/fgbgmask/" + fgbg_folder

      for root_mask, dirs_mask, files_mask in os.walk(path):
          for name_mask in files_mask:
          
             mask_image_list.append(root_mask +'/'+ name_mask)
      path="/content/data/densedepth/" + fgbg_folder

      for root_depth, dirs_depth, files_depth in os.walk(path):
          for name_depth in files_depth:
             depth_image_list.append(root_depth +'/'+ name_depth)
             
  return bg_image_list_final,fgbg_image_list,mask_image_list,depth_image_list            