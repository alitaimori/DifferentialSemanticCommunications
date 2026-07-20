
# Receiver codes of Differential Semantic Communications (DSCs)

"""
Ali Taimori
"""

# clear
# reset -f

try:
    from IPython import get_ipython
    get_ipython().magic('clear') # Clearing Console
    get_ipython().magic('reset -f') # Clearing Variable Explorer
except:
    pass


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
import random
import os



Root_Segmented = "H:/DSC/DSC_Dataset/Segmented/*" # The root of Segmented data
vec_Segmenteds = glob.glob(Root_Segmented)

Root_Text= "H:/DSC/DSC_Dataset/Text/*" # The root of Text data
vec_Texts = glob.glob(Root_Text)


Root_Res = r"H:/DSC/DSC_Dataset/Res/" # The root of results
if not os.path.exists(Root_Res):
    os.makedirs(Root_Res)   

flag_show = 'F' # A true ('T') or false ('F') flag character for showing information


# *********************************
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline

import torch
from PIL import Image, ImageFilter, ImageOps, ImageChops 
import requests
import io
from io import BytesIO

import transformers


pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe = pipe.to("cpu")  # Use CPU if no GPU is available




Root_X_resized = "H:/DSC/DSC_Dataset/X_resized/*" # The root of subjects
vec_Xs = glob.glob(Root_X_resized) # Obtaining the vector of subjects by directory    
N_subjects = len(vec_Xs) # The number of subjects


for k in range(N_subjects): 
    Root_subject_k = vec_Xs[k] + '/*'
    Dir = glob.glob(Root_subject_k) # Directory
    
    def Thumbs_removal(Dir):
        Ni = len(Dir) # Updating the number of images for the subject kth after the Thumbs.db removal 
        flag_thumbs = 'F'
        tmp = 0
        Dir_Thumbs_removed = [] # list initialisation
        for s_thumds in range(Ni):
            file_name = os.path.basename(Dir[s_thumds])
            if file_name != 'Thumbs.db':
                # Dir_Thumbs_removed[tmp] = Dir[s_thumds]
                Dir_Thumbs_removed.append(Dir[s_thumds])
                # Dir_Thumbs_removed.insert(tmp, Dir[s_thumds])
                flag_thumbs = 'T'
                tmp = tmp + 1
        if flag_thumbs == 'T':
            Dir = Dir_Thumbs_removed
        
        # Ni = len(Dir) # Updating the number of images for the subject kth after the Thumbs.db removal 
    
        return Dir
    
    Dir = Thumbs_removal(Dir) # Applying the function Thumbs.db removal 
    Ni = len(Dir) # Updating the number of images for the subject kth after the Thumbs.db removal 
    
    
    Root_Segmented_k = vec_Segmenteds[k] + '/*'
    Dir_Segmented = glob.glob(Root_Segmented_k) # Directory
    Dir_Segmented = Thumbs_removal(Dir_Segmented) # Applying the function Thumbs.db removal 
    
    Root_Text_k = vec_Texts[k] + '/*'
    Dir_Text = glob.glob(Root_Text_k) # Directory
    Dir_Text = Thumbs_removal(Dir_Text) # Applying the function Thumbs.db removal 
    
    
    for i in range(Ni): 


        path = os.path.dirname(Dir[i])
        category_k = os.path.basename(path)
        
        filename = Dir[i]
        filename = os.path.normpath(filename)
        filename = filename.replace("\\", "/")
        image_name = os.path.basename(filename)
        file_name_without_ext = os.path.splitext(image_name)[0]
        
        
        # Load the image and mask.
        with open(Dir[i], "rb") as image_file:
            image = image_file.read()  # Read the image data.

            
        image = Image.open(io.BytesIO(image)) # Convert Bytes to PIL Image.

        image = image.convert('RGB')


        w, h = image.size



        with open(Dir_Segmented[i], "rb") as mask_file:
            mask = mask_file.read()  # Read the image data.
            


        mask = Image.open(io.BytesIO(mask)) # Convert Bytes to PIL Image.


        mask_3Channel = mask.convert('RGB')
        image_masked = ImageChops.multiply(image, mask_3Channel) 
        # image_masked.show() # Show the image.


        mask = mask.convert("1")  # Ensure it is a binary mask.

        mask = ImageOps.invert(mask)
        # mask.show() # inverted mask show

            
        with open(Dir_Text[i], 'r') as file:
            lines = file.readlines()
            row1 = lines[0].strip() if len(lines) > 0 else ''
            row2 = lines[1].strip() if len(lines) > 1 else ''

        def generate_prompt(row1, row2):
            if not row1 and not row2:
                return "Insert a relevant object into a natural background scene."
            elif not row1:
                return f"Insert an appropriate object into a {row2} setting."
            elif not row2:
                return f"Place the {row1} into a plausible scene."
            else:
                return f"Align the {row1} with a {row2}."

        prompt = generate_prompt(row1, row2)


        # Perform inpainting (image modification).
        result = pipe(prompt=prompt, height=h, width=w, image=image_masked, mask_image=mask, num_inference_steps=50)

        # result = pipe(prompt=prompt, image=background_image, mask_image=mask_image, num_inference_steps=50)


        # Show or save the result
        result_image = result.images[0]
        # result_image.show()
        
        Root_Res_plus_category_k = Root_Res + category_k

        str_output = Root_Res_plus_category_k + '/' + file_name_without_ext + '.bmp'

        if not os.path.exists(Root_Res_plus_category_k):
            os.makedirs(Root_Res_plus_category_k)
        # cv2. imwrite(str_output, result_image)
        result_image.save(str_output)
        
        
        
        print(k+1, i+1) # Counters of folder and sub-folder
        
        