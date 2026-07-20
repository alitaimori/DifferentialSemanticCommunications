
# Transmitter codes of Differential Semantic Communications (DSCs)

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:38:59 2024

@author: at3014
"""

try:
    from IPython import get_ipython
    get_ipython().magic('clear') # Clearing Console
    get_ipython().magic('reset -f') # Clearing Variable Explorer
except:
    pass

import cv2
# import imageio.v3 as iio

import torchvision


import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
# import math
import spacy # An open-source software library for advanced natural language processing

import torchvision.models.detection as detection_models
print(dir(detection_models))


from tkinter import *

from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image

flag_show = 'F' # A true ('T') or false ('F') flag character for showing information

# Set pixel threshold for input image resizing
max_pixels = 500_000 # The maximum number of pixels 

Root_Segmented = r"H:/DSC/DSC_Dataset/Segmented/" # The root of segmented images
if not os.path.exists(Root_Segmented):
    os.makedirs(Root_Segmented)

Root_Text = r"H:/DSC/DSC_Dataset/Text/" # The root of text data
if not os.path.exists(Root_Text):
    os.makedirs(Root_Text)
    

Root_Visualisation = r"H:/DSC/DSC_Dataset/Visualisation/" # The root of visualisation of image together with related texts
if not os.path.exists(Root_Visualisation):
    os.makedirs(Root_Visualisation)    

Root_X_resized = r"H:/DSC/DSC_Dataset/X_resized/" # The root of resized input images with preserved aspect ratios
if not os.path.exists(Root_X_resized):
    os.makedirs(Root_X_resized)   

# Physical objects (91 categories)
COCO_dataset_category_name_OriginalPaper = [
    '__background__', # 0 --> environment or context
    'person', # 1
    'bicycle', # 2
    'car', # 3
    'motorcycle', # 4
    'airplane', # 5 
    'bus', # 6
    'train', # 7
    'truck', # 8
    'boat', # 9
    'traffic light', # 10
    'fire hydrant', # 11
    'street sign', # 12 
    'stop sign', # 13
    'parking meter', # 14
    'bench', # 15
    'bird', # 16
    'cat', # 17
    'dog', # 18
    'horse', # 19
    'sheep', # 20
    'cow', # 21
    'elephant', # 22
    'bear', # 23
    'zebra', # 24
    'giraffe', # 25
    'hat', # 26
    'backpack', # 27
    'umbrella', # 28
    'shoe', # 29
    'eye glasses', # 30
	'handbag', # 31
    'tie', # 32
    'suitcase', # 33
    'frisbee', # 34
    'skis', # 35
    'snowboard', # 36
    'sports ball', # 37
    'kite', # 38
    'baseball bat', # 39
    'baseball glove', # 40 
    'skateboard', # 41 
    'surfboard', # 42
    'tennis racket', # 43 
    'bottle', # 44
    'plate', # 45
    'wine glass', # 46
    'cup', # 47
    'fork', # 48
    'knife', # 49
    'spoon', # 50
    'bowl', # 51
    'banana', # 52
    'apple', # 53
    'sandwich', # 54
    'orange', # 55
    'broccoli', # 56
    'carrot', # 57
    'hot dog', # 58
    'pizza', # 59
    'donut', # 60
    'cake', # 61
    'chair', # 62
    'couch', # 63
    'potted plant', # 64
    'bed', # 65
    'mirror', # 66
    'dining table', # 67
    'window', # 68 
    'desk', # 69
    'toilet', # 70
    'door', # 71
    'TV', # 72
    'laptop', # 73
    'mouse', # 74
    'remote', # 75
    'keyboard', # 76
    'cell phone', # 77
    'microwave', # 78
    'oven', # 79
    'toaster', # 80
    'sink', # 81
    'refrigerator', # 82
    'blender', # 83
    'book', # 84
    'clock', # 85
    'vase', # 86
    'scissors', # 87
    'teddy bear', # 88
    'hair drier', # 89
    'toothbrush', # 90
    'hair brush' # 91	
    ]

N_objects = len(COCO_dataset_category_name_OriginalPaper) - 1 # We exclude the background.


# The list of default contexts for the 91 COCO objects.
# We use this list when we are not able to extract any context from a given image.
default_context = [
'street', # 1  
'road', # 2  
'road', # 3  
'road', # 4  
'sky', # 5  
'station', # 6  
'railway', # 7  
'highway', # 8  
'water', # 9   
'intersection', # 10  
'pavement', # 11  
'roadside', # 12  
'roadside', # 13  
'parking', # 14  
'park', # 15  
'sky', # 16  
'home', # 17  
'yard', # 18  
'farm', # 19  
'pasture', # 20  
'farm', # 21  
'savanna', # 22  
'forest', # 23  
'savanna', # 24  
'savanna', # 25  
'wardrobe', # 26  
'school', # 27  
'rain', # 28  
'footpath', # 29 
'face', # 30  
'shopping', # 31  
'office', # 32  
'airport', # 33  
'park', # 34  
'mountain', # 35  
'slope', # 36  
'field', # 37  
'park', # 38  
'stadium', # 39  
'stadium', # 40  
'skatepark', # 41  
'beach', # 42 
'court', # 43  
'table', # 44  
'kitchen', # 45  
'restaurant', # 46  
'table', # 47  
'kitchen', # 48  
'kitchen', # 49  
'kitchen', # 50  
'kitchen', # 51  
'market', # 52 
'orchard', # 53  
'cafeteria', # 54  
'orchard', # 55  
'market', # 56  
'farm', # 57 
'street', # 58  
'restaurant', # 59  
'bakery', # 60  
'bakery', # 61  
'room', # 62  
'livingroom', # 63  
'balcony', # 64  
'bedroom', # 65 
'bathroom', # 66 
'diningroom', # 67  
'building', # 68  
'office', # 69 
'restroom', # 70 
'entrance', # 71 
'livingroom', # 72 
'office', # 73 
'desk', # 74 
'livingroom', # 75  
'desk', # 76 
'pocket', # 77 
'kitchen', # 78 
'kitchen', # 79 
'kitchen', # 80 
'kitchen', # 81  
'kitchen', # 82 
'kitchen', # 83 
'library', # 84 
'wall', # 85 
'table', # 86 
'desk', # 87 
'bedroom', # 88 
'bathroom', # 89 
'bathroom', # 90 
'bathroom', # 91 
]


# Load the pre-trained Panoptic FPN model
# model = models.detection.panoptic_fpn_resnet50(pretrained=True)
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True) # Mask Region-based Convolutional Neural Network (Mask R-CNN)

model.eval()  # Set the model to evaluation mode.



# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_BLIP = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



Root_X = "H:/DSC/DSC_Dataset/X/*" # The root of subjects
vec_Xs = glob.glob(Root_X) # Obtaining the vector of subjects by directory    
N_subjects = len(vec_Xs) # The number of subjects

for k in range(N_subjects):
    Root_subject_k = vec_Xs[k] + '/*'
    Dir = glob.glob(Root_subject_k) # directory
    
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
    
    for i in range(Ni):
# =============================================================================
#         file_extension = os.path.splitext(Dir[i])[1]
#         if file_extension == '.avif':
#             input_image = iio.imread(Dir[i]) # Read the AVIF file, which is a RGB image
#             input_image_BGR = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
#         else:
#             input_image_BGR = cv2.imread(Dir[i], cv2.IMREAD_COLOR) # foreground plus background image
# =============================================================================
        input_image_BGR = cv2.imread(Dir[i], cv2.IMREAD_COLOR) # foreground plus background image
        h, w, z = input_image_BGR.shape # image size
        if flag_show == 'T':
            cv2.imshow('X', input_image_BGR) # Creating the window
            cv2.waitKey(0) # Showing the image --> It waits for user to press any key; this is necessary to avoid Python kernel from crashing.
            cv2.destroyAllWindows() # Closing all open windows


        path = os.path.dirname(Dir[i])
        category_k = os.path.basename(path)
        
        filename = Dir[i]
        filename = os.path.normpath(filename)
        filename = filename.replace("\\", "/")
        image_name = os.path.basename(filename)
        file_name_without_ext = os.path.splitext(image_name)[0]

        
        # *********************************************************************
        # ************************ Resizing input image ***********************
        original_pixels = h * w
        # Check if resizing is needed
        # if original_pixels > max_pixels:
        scale_factor = (max_pixels / original_pixels) ** 0.5  # sqrt to apply scale to both dimensions
        # w = int(w * scale_factor)
        # h = int(h * scale_factor)
        w = round((w * scale_factor)/8)*8
        h = round((h * scale_factor)/8)*8
        
        interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
        # Resize the image
        input_image_BGR = cv2.resize(input_image_BGR, (w, h), interpolation = interpolation)
        
        Root_X_resized_plus_category_k = Root_X_resized + category_k

        str_output = Root_X_resized_plus_category_k + '/' + file_name_without_ext + '.bmp'

        if not os.path.exists(Root_X_resized_plus_category_k):
            os.makedirs(Root_X_resized_plus_category_k)
        cv2. imwrite(str_output, input_image_BGR)
        # ******************** End of resizing input image ********************
        # ********************************************************************* 


        # Load and preprocess the image
        
        
        # ******************* Resizing the image to the screen size *******************
        # Create an instance of tkinter frame.
        win= Tk()
        
        # Set the geometry of frame.
        win.geometry("650x250")
        
        # Get the current screen width and height.
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        
        # Print the screen size.
        print("Screen width:", screen_width)
        print("Screen height:", screen_height)
        
        # input_image_BGR = cv2.resize(input_image_BGR, (screen_width, screen_height), interpolation=cv2.INTER_AREA) # (768, 576)
        # *****************************************************************************
        
        # X_not = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2GRAY) # Grey-scale image
        input_image = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2RGB) # Converting to RGB
        
        
        # ************************************ BLIP ***********************************
        # Process and generate caption.
        inputs_BLIP = processor(input_image, return_tensors="pt")
        # inputs_BLIP = processor(input_image, return_tensors="pt", text="Describe the main objects:")
        # output_BLIP = model_BLIP.generate(**inputs_BLIP)
        output_BLIP = model_BLIP.generate(**inputs_BLIP,
            max_length = 15,  # Limit caption length (avoid excessive detail).
            min_length = 3,    # Ensure at least 3 tokens.
            num_beams = 3,  # Improve structured output.
            length_penalty = 0.5  # Slightly penalises long captions.
            #temperature = 0.7,  # Balance between deterministic and diverse results.
            #top_p = 0.9  # Focus on high-probability words, avoiding unnecessary randomness.
        )
        caption = processor.decode(output_BLIP[0], skip_special_tokens=True)
        print("Caption:", caption)
        # *****************************************************************************
        
        # ************** context extraction from image's caption **************
        # import spacy

        nlp = spacy.load("en_core_web_sm") # Load the English spaCy model.
        doc_tokenised_caption = nlp(caption)
        
        
        # *********************************************************************
        # *********************** Captioning bug control **********************
        # This code segment is to control bugs of the caption. For example, the object 'teady bear' is rejected during nouns filtering. So, we use a trick to retrieve it.
        # Note that we may use this code segment to control other exceptions right after the caption extreaction.
        mycaption = caption.split() # Converting str to list
        # L_words = len(doc_tokenised_caption)
        L_words = len(mycaption) 
        words_unified = [] # Initialisation of a dynamic list for unification of labels with two or more words
        tmp = 0
        while tmp < L_words:
            current_word = mycaption[tmp]
            
            next_word = '' # Initialisation
            if tmp < L_words - 1:
                next_word = mycaption[tmp + 1] 
                
            if current_word == 'oven':
                concatenated_words = 'ovenequipment' # This is a trick for preserving 'oven'.
                words_unified.append(concatenated_words)
                tmp += 1
                continue
            
            if current_word == 'scoot' or current_word == 'scoote':
                concatenated_words = 'scooter'
                words_unified.append(concatenated_words)
                tmp += 1
                continue
            
            if current_word == 'teddy' and next_word == 'bear':
                concatenated_words = 'teddybear'
                words_unified.append(concatenated_words)
                tmp += 2
                continue
            
            if (current_word == 'person' and next_word == 'holding') or (current_word == 'man' and next_word == 'holding') or (current_word == 'woman' and next_word == 'holding') or (current_word == 'boy' and next_word == 'holding') or (current_word == 'girl' and next_word == 'holding') or (current_word == 'daughter' and next_word == 'holding') or (current_word == 'child' and next_word == 'holding') or (current_word == 'kid' and next_word == 'holding') or (current_word == 'baby' and next_word == 'holding') or (current_word == 'hand' and next_word == 'holding'):
                concatenated_words = 'holding' 
                words_unified.append(concatenated_words)
                tmp += 2
                continue

            if (current_word == 'person' and next_word == 'cutting') or (current_word == 'hand' and next_word == 'cutting'):
                concatenated_words = 'cutting' 
                words_unified.append(concatenated_words)
                tmp += 2
                continue
                
            # If no condition is met, just add the noun as is.
            words_unified.append(current_word)
            tmp += 1
            
        words_unified_list2str = " ".join(words_unified) # Converting list to str
        doc_tokenised_caption = nlp(words_unified_list2str)
        # doc_tokenised_caption = words_unified
        # *********************************************************************
        # *********************************************************************
        
        

        # ************************* Removing compounds ************************
        # Descriptive compound nouns
        # A compound is a noun, an adjective or a verb made of two or more words or parts of words, written as one or more words, or joined by a hyphen. Travel agent, dark-haired and bathroom are all compounds.
        # Expandable set of descriptive nouns (materials, attributes, origins, etc.)
        descriptive_nouns = {"stone", "wood", "metal", "plastic", "glass", "paper", "gold", "silver", "chrome",
                             "rubber", "cloth", "concrete", "brick", "ceramic", "marble", "wooden", "leather", "round"}
        
        def remove_descriptive_nouns(doc):
            # doc = nlp(text)
            tokens_to_remove = set()
        
            for token in doc:
                # Remove any token that is in "descriptive_nouns". Identify descriptive nouns in compound structures.
                if token.text.lower() in descriptive_nouns: # token.dep_ == "compound" and token.text.lower() in descriptive_nouns
                    tokens_to_remove.add(token.i)  # Mark for removal.
        
            # Reconstruct sentence without the marked words.
            filtered_tokens = [token.text for i, token in enumerate(doc) if i not in tokens_to_remove]
            
            return " ".join(filtered_tokens)
        
        # Example usage
        # text = "a wooden bench sitting in front of a stone wall"
        filtered_compound = remove_descriptive_nouns(doc_tokenised_caption)
        print("Ali's print-", filtered_compound)  # Output: "a bench sitting in front of a wall"

        doc_tokenised_filtered_compound = nlp(filtered_compound)
        # *********************************************************************
        
        # Extract nouns.
        nouns = [token.text for token in doc_tokenised_filtered_compound if token.pos_ == "NOUN"]
        print(nouns)
        
        
        # ************************ Unification of nouns ***********************
        # nouns.append(nouns[-1]) # Padding at the end of string
        L_nouns = len(nouns)
        nouns_unified = [] # Initialisation of a dynamic list for unification of labels with two or more words
        tmp = 0
        # flag_2words = 'False'
        while tmp < L_nouns:
            # if tmp < L_nouns - 1:
            current_noun = nouns[tmp]
            
            next_noun = '' # Initialisation
            if tmp < L_nouns - 1:
                next_noun = nouns[tmp + 1] 
            
            if current_noun == 'baseball' and next_noun == 'bat':
                concatenated_nouns = 'baseball' + ' ' + 'bat'
                nouns_unified.append(concatenated_nouns)
                tmp += 2 # Skip the next noun as it's already unified.
                # flag_2words = 'True'
                continue # The continue statement inside the if statement is used to skip the remaining code in the current iteration of the loop and move to the next iteration. 
            
            if current_noun == 'baseball' and next_noun == 'glove':
                concatenated_nouns = 'baseball' + ' ' + 'glove'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'cell' and next_noun == 'phone':
                concatenated_nouns = 'cell' + ' ' + 'phone'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'dining' and next_noun == 'table':
                concatenated_nouns = 'dining' + ' ' + 'table'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'eye' and next_noun == 'glasses':
                concatenated_nouns = 'eye' + ' ' + 'glasses'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'fire' and next_noun == 'hydrant':
                concatenated_nouns = 'fire' + ' ' + 'hydrant'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'hair' and next_noun == 'brush':
                concatenated_nouns = 'hair' + ' ' + 'brush'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'hair' and next_noun == 'drier') or (current_noun == 'hair' and next_noun == 'dryer'):
                concatenated_nouns = 'hair' + ' ' + 'drier'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'hot' and next_noun == 'dog':
                concatenated_nouns = 'hot' + ' ' + 'dog'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'parking' and next_noun == 'meter':
                concatenated_nouns = 'parking' + ' ' + 'meter'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'potted' and next_noun == 'plant':
                concatenated_nouns = 'potted' + ' ' + 'plant'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'refrigerator' and next_noun == 'freezer':
                concatenated_nouns = 'refrigerator'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'sports' and next_noun == 'ball':
                concatenated_nouns = 'sports' + ' ' + 'ball'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'stop' and next_noun == 'sign':
                concatenated_nouns = 'stop' + ' ' + 'sign'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'street' and next_noun == 'sign':
                concatenated_nouns = 'street' + ' ' + 'sign'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'teddy' and next_noun == 'bear':
                concatenated_nouns = 'teddy' + ' ' + 'bear'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'tennis' and next_noun == 'racket':
                concatenated_nouns = 'tennis' + ' ' + 'racket'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'traffic' and next_noun == 'light':
                concatenated_nouns = 'traffic' + ' ' + 'light'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'wine' and next_noun == 'glass':
                concatenated_nouns = 'wine' + ' ' + 'glass'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            # For nouns related to the performance of captioning
            if current_noun == 'pair' and next_noun == 'glasses':
                concatenated_nouns = 'eye' + ' ' + 'glasses'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'pair' and next_noun == 'scissors':
                concatenated_nouns = 'scissors'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'pair' and next_noun == 'shoes':
                concatenated_nouns = 'shoe'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'pair' and next_noun == 'skis': # Note: "skis" is the plural form of "ski".
                concatenated_nouns = 'skis'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'tooth' and next_noun == 'brush':
                concatenated_nouns = 'toothbrush'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            # In the following, we categorise all types of women bags in the class of 'handbag'. Note that this mayn't be an accurate categorisation of women bags. For example, a 'shoulder bag' is not necessarily a 'handbag'.
            if (current_noun == 'purse' and next_noun == 'bag') or (current_noun == 'crossbody' and next_noun == 'bag') or (current_noun == 'cowhide' and next_noun == 'bag') or (current_noun == 'shoulder' and next_noun == 'bag'):
                concatenated_nouns = 'handbag'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'park' and next_noun == 'bench':
                concatenated_nouns = 'bench'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'computer' and next_noun == 'desk':
                concatenated_nouns = 'desk'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'laptop' and next_noun =='computer') or (current_noun == 'computer' and next_noun =='laptop'):
                concatenated_nouns = 'laptop'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'computer' and next_noun =='mouse') or (current_noun == 'laptop' and next_noun =='mouse'):
                concatenated_nouns = 'mouse'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'computer' and next_noun =='keyboard') or (current_noun == 'laptop' and next_noun =='keyboard'):
                concatenated_nouns = 'keyboard'
                nouns_unified.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'baby':
                import nltk
                nltk.download('wordnet')  # For WordNet (animal classification)
                nltk.download('omw-1.4')  # Optional: Improves WordNet with more languages
                from nltk.corpus import wordnet # WordNet's lexical database
                
                def is_animal_wordnet(noun):
                    """Check if a noun is classified as an animal in WordNet."""
                    for synset in wordnet.synsets(noun, pos=wordnet.NOUN):
                        if "animal" in synset.lexname():  # WordNet categorizes animals under 'animal.n.01'
                            return True
                    return False
                
                flag = is_animal_wordnet(next_noun)
                if flag == True:
                    concatenated_nouns = next_noun
                    nouns_unified.append(concatenated_nouns)
                    # print("Ali- Flag is true.") # This line is for reaching inside this "if statement".
                    tmp += 2
                    continue

                
            # If no condition is met, just add the noun as is.
            nouns_unified.append(current_noun)
            tmp += 1
         
        # *********************************************************************
                    
                
        # ****** Converting nouns to the standard forms of COCO's objects ******
        L_nouns_unified = len(nouns_unified)
        if L_nouns_unified >= 1: 
            # nouns_first_half_part = round(L_nouns_unified/2) 
            for noun in range(L_nouns_unified): # range(nouns_first_half_part): 
                my_noun = nouns_unified[noun]
                
                # *** Standardisation of nouns to the COCO object categories **
                if my_noun == 'woman' or my_noun == 'girl' or my_noun == 'daughter' or my_noun == 'man' or my_noun == 'boy' or my_noun == 'child' or my_noun == 'kid' or my_noun == 'baby':
                    nouns_unified[noun] = 'person'
                    
                if my_noun == 'bike':
                    nouns_unified[noun] = 'bicycle'

                if my_noun == 'taxi':
                    nouns_unified[noun] = 'car'
                    
                if my_noun == 'wine':
                    nouns_unified[noun] = 'wine glass'
                    
                if my_noun == 'plane' or my_noun == 'aeroplane': 
                    nouns_unified[noun] = 'airplane'
                    
                if my_noun == 'gate': # The object 'gate' is a big 'door'. Thus, we consider 'gate' as a sub-class of 'door'.
                    nouns_unified[noun] = 'door'
                    
                if my_noun == 'cell' or my_noun == 'mobile' or my_noun == 'iPhone' or my_noun == 'iphone': 
                    nouns_unified[noun] = 'cell phone'
                    
                if my_noun == 'tv' or my_noun == 'television': 
                    nouns_unified[noun] = 'TV'
                    
                if my_noun == 'hairdrier' or my_noun == 'hairdryer': 
                    nouns_unified[noun] = 'hair drier'
                    
                if my_noun == 'brush': 
                    nouns_unified[noun] = 'hair brush'
                    
                if my_noun == 'bag': 
                    nouns_unified[noun] = 'handbag'
                    
                if my_noun == 'purse': 
                    nouns_unified[noun] = 'handbag'
                    
                if my_noun == 'lamb': 
                    nouns_unified[noun] = 'sheep'
                    
                if my_noun == 'kitten': # Note that kitten is a young cat.
                    nouns_unified[noun] = 'cat'
                    
                if my_noun == 'teddybear': 
                    nouns_unified[noun] = 'teddy bear'
                    
                if my_noun == 'ovenequipment':
                    nouns_unified[noun] = 'oven'
                    
                if my_noun == 'cap': # Note that 'cap' is a type or sub-class of 'hat'.
                    nouns_unified[noun] = 'hat'
                    
                if my_noun == 'eagle' or my_noun == 'swan' or my_noun == 'duck' or my_noun == 'goose': # Sub-classes of bird. You can expand it. 
                    nouns_unified[noun] = 'bird'
                    
                if my_noun == 'broccole' or my_noun == 'broccote' or my_noun == 'brocco': # The grammer of broccole is incorrect. Also, note that the word broccoli is an nucontable noun. 
                    nouns_unified[noun] = 'broccoli'                   

                if my_noun == 'gife' or my_noun == 'gi' or my_noun == 'git': # The grammer of gife is incorrect. 
                    nouns_unified[noun] = 'giraffe'   
                    
                if my_noun == 'fr' or my_noun == 'Frisbee': # The letters "fr" seems to be a bug in caption extraction.
                    nouns_unified[noun] = 'frisbee' 
                    
                if my_noun == 'dog':
                    indices = [token.i for token in doc_tokenised_caption if token.text.lower() == "dog"]
                    min_index = min(indices)
                    if min_index > 0:
                        dog_token = doc_tokenised_caption[min_index - 1]  # Get the spaCy token.
                        dog_text = caption[dog_token.idx : dog_token.idx + len(dog_token.text)]  # Extract substring from original text.
                        if dog_text == 'hot': # doc_tokenised_caption[min_index - 1] == 'hot':
                            nouns_unified[noun] = 'hot dog'

                            
        # *********************************************************************
            
        
        
        # object_label = COCO_dataset_category_name_OriginalPaper[labels[0]] # Object labelling automatically 
        
        # background_label = 'farm'. Background labelling by hand 
        background_label = " ".join(nouns_unified)  # Joins with a space. Background labelling automatically
        
        
        
        # ******************* Removing nouns with repetition ****************** 
        # The function "remove_duplicate_words", which works based on spaCy NLP library, removes the nouns that have repetition, e.g., two times or more. It just preserves one of them. 
        def remove_duplicate_words(text):
            # Tokenize text with spaCy
            doc = nlp(text)
            
            # Track seen words while preserving order.
            seen_words = set()
            filtered_tokens = []
            
            for token in doc:
                lower_word = token.text.lower()
                
                # If it's a word and not seen before, keep it.
                if token.is_alpha and lower_word not in seen_words:
                    seen_words.add(lower_word)
                    filtered_tokens.append(token.text)
                elif not token.is_alpha:
                    # Keep punctuation & spaces.
                    filtered_tokens.append(token.text)
            
            # Reconstruct the cleaned text
            cleaned_text = " ".join(filtered_tokens).strip()
            
            return cleaned_text
            # return cleaned_text.strip()
        
        background_label = remove_duplicate_words(background_label) # Cleaned background label
        # print(cleaned_text)
        # *********************************************************************
        
        
        
        
        # ************************** Removing gerunds ************************* 
        # A gerund is a type of noun.
        # A gerund is a noun in the form of the present participle of a verb (that is, ending in -ing), for example "travelling" in the sentence: I preferred travelling alone.
        # gerunds = {"swimming", "sailing", "standing", "biking", "running", "walking", "jumping", "dancing"} # Predefined list of gerunds. This is an example set. So, you can expand it as follows.
        # Here is the list of all important A to Z gerunds:
        gerunds = [
        'accepting', # 1
        'accomplishing', # 2     
        'accounting', # 3 
        'achiving', # 4        
        'acting', # 5        
        'adding', # 6         
        'admiring', # 7         
        'advising', # 8         
        'affording', # 9        
        'allowing', # 10        
        'announcing', # 11        
        'answering', # 12         
        'apologizing', # 13         
        'appearing', # 14         
        'appreciating', # 15         
        'approaching', # 16         
        'arranging', # 17         
        'asking', # 18        
        'attaching', # 19         
        'attending', # 20        
        'avoiding', # 21         
        'backing', # 22       
        'balancing', # 23        
        'bargaining', # 24         
        'bashing', # 25       
        'battling', # 26      
        'becoming', # 27       
        'beginning', # 28         
        'behaving', # 29       
        'believing', # 30  
        'biking',
        'blaming', # 31      
        'blessing', # 32      
        'boasting', # 33       
        'booking', # 34      
        'breathing', # 35        
        'calling', # 36       
        'carrying', # 37       
        'catching', # 38       
        'challenging', # 39       
        'changing', # 40        
        'checking', # 41       
        'choking', # 42        
        'choosing', # 43        
        'claiming', # 44       
        'closing', # 45       
        'collecting', # 46         
        'comforting', # 47        
        'communicating', # 48        
        'comparing', # 49        
        'complaining', # 50         
        'completing', # 51       
        'concentrating', # 52        
        'considering', # 53     
        'consisting', # 54     
        'consuming', # 55   
        'containing', # 56      
        'continuing', # 57 
        'cooking', # 
        'copying', # 58        
        'correcting', # 59        
        'costing', # 60         
        'counting', # 61         
        'covering', # 62        
        'cracking', # 63        
        'creating', # 64        
        'crossing', # 65        
        'crying', # 66      
        'curling', # 67         
        'cutting', # 68   
        'dancing',
        'deciding', # 69        
        'declaring', # 70        
        'decorating', # 71         
        'delivering', # 72        
        'demanding', # 73        
        'describing', # 74       
        'designing', # 75        
        'destroying', # 76        
        'developing', # 77       
        'disappearing', # 78         
        'discovering', # 79        
        'discussing', # 80         
        'disobeying', # 81        
        'doing', # 82         
        'doubting', # 83        
        'drawing', # 84        
        'dreaming', # 85         
        'drinking', # 86         
        'driving', # 87        
        'eating', # 88         
        'enjoying', # 89         
        'entertaining', # 90         
        'escaping', # 91        
        'examining', # 92        
        'existing', # 93         
        'expecting', # 94         
        'explaining', # 95         
        'exploring', # 96        
        'expressing', # 97        
        'farming', # 98       
        'fetching', # 99       
        'fighting', # 100         
        'filling', # 101         
        'finding', # 102       
        'finishing', # 103        
        'following', # 104        
        'forbidding', # 105         
        'forgiving', # 106       
        'forming', # 107      
        'founding', # 108        
        'gaining', # 109       
        'gathering', # 110        
        'getting', # 111       
        'giving', # 112     
        'going', # 113     
        'greeting', # 114       
        'growing', # 115      
        'guessing', # 116      
        'handling', # 117      
        'hanging', # 118     
        'happening', # 119       
        'helping', # 120        
        'hiding', # 121      
        'holding', # 122       
        'hoping', # 123        
        'hurting', # 124       
        'identifying', # 125         
        'imagining', # 126       
        'importing', # 127       
        'including', # 128        
        'increasing', # 129         
        'informing', # 130       
        'inserting', # 131        
        'inspiring', # 132        
        'instructing', # 133        
        'inviting', # 134        
        'joining', # 135 
        'jumping', 
        'keeping', # 136        
        'kicking', # 137         
        'killing', # 138 
        'knitting', #         
        'knowing', # 139        
        'lasting', # 140         
        'laughing', # 141         
        'leading', # 142         
        'listening', # 143         
        'living', # 144         
        'locking', # 145         
        'looking', # 146         
        'loving', # 147        
        'managing', # 148         
        'marrying', # 149         
        'matching', # 150         
        'measuring', # 151         
        'meeting', # 152        
        'memorizing', # 153         
        'mentioning', # 154        
        'mixing', # 155         
        'moving', # 156        
        'naming', # 157         
        'needing', # 158        
        'noticing', # 159         
        'obtaining', # 160         
        'occuring', # 161       
        'offering', # 162        
        'operating', # 163         
        'ordering', # 164         
        'organizing', # 165         
        'originating', # 166         
        'overcoming', # 167      
        'painting', # 
        'paying', # 168         
        'performing', # 169        
        'planning', # 170         
        'playing', # 171         
        'pointing', # 172        
        'possessing', # 173         
        'posting', # 174        
        'practising', # 175         
        'preaching', # 176         
        'picking', # 177         
        'placing', # 178         
        'planting', # 179         
        'pointing', # 180        
        'presenting', # 181         
        'preserving', # 182        
        'pretending', # 183         
        'preventing', # 184        
        'producing', # 185        
        'promising', # 186         
        'protecting', # 187         
        'providing', # 188         
        'pulling', # 189         
        'pushing', # 190         
        'questioning', # 191       
        'reaching', # 192       
        'reacting', # 193       
        'reading', # 194     
        'realizing', # 195       
        'receiving', # 196       
        'recognizing', # 197        
        'recommending', # 198        
        'recording', # 199      
        'recovering', # 200       
        'refusing', # 201     
        'regretting', # 202       
        'rejecting', # 203      
        'rejoicing', # 204       
        'remaining', # 205       
        'remembering', # 206        
        'repeating', # 207        
        'replacing', # 208        
        'reporting', # 209        
        'representing', # 210        
        'requesting', # 211      
        'rescuing', # 212       
        'researching', # 213        
        'responding', # 214       
        'resting', # 215       
        'retaining', # 216       
        'returning', # 217        
        'ruling', # 218       
        'running', # 219      
        'sailing', 
        'screaming', # 220        
        'searching', # 221        
        'seeing', # 222      
        'selling', # 223       
        'serving', # 224        
        'setting', # 225       
        'shaking', # 226        
        'sharing', # 227       
        'showing', # 228        
        'singing', # 229        
        'smiling', # 230       
        'solving', # 231       
        'speaking', # 232        
        'splitting', # 233        
        'spreading', # 234 
        'standing',
        'starting', # 235       
        'staying', # 236      
        'stepping', # 237       
        'stopping', # 238      
        'struggling', # 239       
        'suggesting', # 240       
        'supplying', # 241       
        'supporting', # 242       
        'suspecting', # 243  
        'swimming',
        'taking', # 244      
        'teaching', # 245       
        'telling', # 246     
        'testing', # 247     
        'thinking', # 248      
        'throwing', # 249      
        'touching', # 250     
        'training', # 251     
        'travelling', # 252       
        'treating', # 253       
        'trusting', # 254      
        'trying', # 255     
        'turning', # 256     
        'using', # 257   
        'visiting', # 258     
        'waiting', # 259    
        'waking', # 260    
        'walking', # 261     
        'wanting', # 262     
        'warning', # 263    
        'watching', # 264   
        'weeping', # 265  
        'winning', # 266    
        'wishing', # 267    
        'working', # 268    
        'worrying', # 269     
        'writing', # 270    
        'yelling', # 271     
        'zipping', # 272     
        'zooming' # 273
        ]
  
        # Split text into words, filter out gerunds, and reconstruct
        background_label = " ".join(word for word in background_label.split() if word.lower() not in gerunds)
        # *********************************************************************
        
        # *********************************************************************
        # ******************* Removing non-physical objects *******************
        # In this code segment, we want to remove nouns with non-physical sense (intangible entities).
        # Example:
        # nonphysical_nouns = [
        #     'top',
        #     'front',
        #     'side',
        #     'background',
        #     'middle',
        #     'piece',
        #     'display',
        #     'design',
        #     'lots',
        #     'pattern'
        #     ]
        nonphysical_nouns = [
            # Spatial & Positional Terms
            'top', 'bottom', 'front', 'back', 'side', 'left', 'right', 
            'center', 'middle', 'edge', 'corner', 'background', 'foreground',
        
            # Abstract Parts & Descriptions
            'section', 'portion', 'piece', 'part', 'segment', 'area', 'region', 
            'shape', 'form', 'outline', 'figure',
        
            # Visual & Design Elements
            'display', 'design', 'pattern', 'image', 'view', 'scene', 
            'picture', 'appearance', 'vision', 'representation',
        
            # Quantifiers & Abstract Amounts
            'lots', 'group', 'collection', 'mass', 'bulk', 'series', 
            'set', 'array', 'pile', 'heap', 'cluster', 'bunch', 'stack',
        
            # Conceptual & Functional Terms
            'symbol', 'icon', 'mark', 'feature', 'aspect', 
            'concept', 'idea', 'notion', 'element', 'property', # sign,
        
            # Measurement & Scale Terms
            'size', 'length', 'width', 'height', 'volume', 'depth', 
            'proportion', 'ratio', 'dimension', 'scale', 'level',
            
            # Other Miscellaneous Ones
            'close', 'distance', 'position', 'perspective', 'focus', 'structure', 'composition', 
            'shadow', 'shade', 'reflection', 'glow', 'contrast', 'saturation', 'tone', 'alignment',
            'symmetry', 'balance', 'proximity', 'order', 'relation', 'flight', 'hind', 'wireless'   
        ]
        # Method 1:
        # background_label = " ".join(word for word in background_label.split() if word.lower() not in nonphysical_nouns) # Method 1: This just removes non-plural nouns.
    
        # Method 2:    
        from nltk.stem import PorterStemmer # Stemmed non-physical nouns list
        stemmer = PorterStemmer() # Initialise stemmer.
        nonphysical_nouns = set(stemmer.stem(word) for word in nonphysical_nouns)
        # Process the background_label as follows:
        background_label = " ".join(word for word in background_label.split() if stemmer.stem(word.lower()) not in nonphysical_nouns) # Method 2: This removes both non-plural (stem) and plural nouns.                                
        # *********************************************************************
        # *********************************************************************


                
        # * Removing object/context's sub-classes or reordering object's parts
        # Removing descriptive sub-classes of a context or correcting the order of an object
        # Example 1: sail boat --> boat
        # Example 2: field of grass --> field Note: Here, "grass" is a sub-class for "field", which provides more details about the field.
        # Example 3: glass of wine --> wine glass
        # The programmer can expand the above exceptions by adding more if statements in the following codes.
        candidate_nouns = background_label.split()
        # candidate_nouns = " ".join(candidate_nouns)  # Joins with a space 

        L_nouns = len(candidate_nouns)
        background_label_subcalss_removed = [] # Initialisation of a dynamic list for removing the above redundancies
        tmp = 0
        while tmp < L_nouns:
            current_noun = candidate_nouns[tmp]
            
            next_noun = '' # Initialisation
            if tmp < L_nouns - 1:
                next_noun = candidate_nouns[tmp + 1] 

            if current_noun == 'sail' and next_noun == 'boat':
                concatenated_nouns = 'boat'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            if (current_noun == 'taxi' and next_noun == 'car') or (current_noun == 'pickup' and next_noun == 'truck') or (current_noun == 'police' and next_noun == 'car'):
                concatenated_nouns = 'car'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'baseball' and next_noun == 'hat') or (current_noun == 'hard' and next_noun == 'hat') or (current_noun == 'tweed' and next_noun == 'hat') or (current_noun == 'bobble' and next_noun == 'hat'):
                concatenated_nouns = 'hat'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'decker' and next_noun == 'bus':
                concatenated_nouns = 'bus'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            if (current_noun == 'road' and next_noun == 'sign') or (current_noun == 'way' and next_noun == 'sign') or (current_noun == 'parking' and next_noun == 'sign'):
                concatenated_nouns = 'street' + ' ' + 'sign'
                background_label_subcalss_removed.append(concatenated_nouns)
                # current_noun = 'street'
                tmp += 2
                continue
                  
            if current_noun == 'field' and next_noun == 'grass':
                concatenated_nouns = 'field'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'field' and next_noun == 'plants':
                concatenated_nouns = 'field'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue 
            
            if current_noun == 'field' and next_noun == 'flowers':
                concatenated_nouns = 'field'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'field' and next_noun == 'sunflowers':
                concatenated_nouns = 'field'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'body' and next_noun == 'water':
                concatenated_nouns = 'water'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'glass' and next_noun == 'wine':
                concatenated_nouns = 'wine' + ' ' + 'glass'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'tree' and next_noun == 'branch':
                concatenated_nouns = 'tree'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'hotel' and next_noun == 'room') or (current_noun == 'house' and next_noun == 'room'):
                concatenated_nouns = 'room'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'dirt' and next_noun == 'road') or (current_noun == 'sand' and next_noun == 'road') or (current_noun == 'asphalt' and next_noun == 'road'):
                concatenated_nouns = 'road'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue     
            
            if (current_noun == 'kitchen' and next_noun == 'counter') or (current_noun == 'bar' and next_noun == 'counter') or (current_noun == 'airport' and next_noun == 'counter'):
                concatenated_nouns = 'counter'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'water' and next_noun == 'bottle') or (current_noun == 'bottle' and next_noun == 'water') or (current_noun == 'juice' and next_noun == 'bottle') or (current_noun == 'bottle' and next_noun == 'juice') or (current_noun == 'wine' and next_noun == 'bottle') or (current_noun == 'bottle' and next_noun == 'wine') or (current_noun == 'whiskey' and next_noun == 'bottle') or (current_noun == 'bottle' and next_noun == 'whiskey') or (current_noun == 'whisky' and next_noun == 'bottle') or (current_noun == 'bottle' and next_noun == 'whisky'):
                concatenated_nouns = 'bottle'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'bowl' and next_noun == 'soup') or (current_noun == 'soup' and next_noun == 'bowl'):
                concatenated_nouns = 'bowl'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'cup' and next_noun == 'coffee') or (current_noun == 'cup' and next_noun == 'tea'):
                concatenated_nouns = 'cup'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            if (current_noun == 'wall' and next_noun == 'room') or (current_noun == 'wall' and next_noun == 'yard'):
                concatenated_nouns = 'wall'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            if (current_noun == 'office' and next_noun == 'desk') or (current_noun == 'computer' and next_noun == 'desk') or (current_noun == 'libray' and next_noun == 'desk'):
                concatenated_nouns = 'desk'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'microwave' and next_noun == 'oven': # Note: the noun "microwave" is a type of oven, but we have the label of this sub-class. Thus, we want to identify it, regardless of "oven". Note also that "microwave oven" is a formal name for "microwave".
                concatenated_nouns = 'microwave'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if current_noun == 'water' and next_noun == 'river':
                concatenated_nouns = 'river'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            if current_noun == 'plant' and next_noun == 'pot':
                concatenated_nouns = 'potted plant'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue
            
            if (current_noun == 'soccer' and next_noun == 'ball') or (current_noun == 'football' and next_noun == 'ball') or (current_noun == 'volleyball' and next_noun == 'ball') or (current_noun == 'basketball' and next_noun == 'ball') or (current_noun == 'hockey' and next_noun == 'ball') or (current_noun == 'tennis' and next_noun == 'ball') or (current_noun == 'ping-pong' and next_noun == 'ball') or (current_noun == 'Ping-Pong' and next_noun == 'ball') or (current_noun == 'golf' and next_noun == 'ball'):
                concatenated_nouns = 'ball'
                background_label_subcalss_removed.append(concatenated_nouns)
                tmp += 2
                continue

            # If no condition is met, just add the noun as is.
            background_label_subcalss_removed.append(current_noun)
            tmp += 1
            # *****************************************************************   
        

        # ************************ Correction of 'sign' ***********************
        # Specific correction
        L_object_context = len(background_label_subcalss_removed)
        for jj in range(L_object_context):
            name_check = background_label_subcalss_removed[jj]
            if jj == 0 and name_check == 'sign': # The first word is 'sign'.
                background_label_subcalss_removed[jj] = 'street sign'
            elif jj != 0 and name_check == 'sign': # A word after the first one is 'sign'.
                if background_label_subcalss_removed[jj - 1] != 'stop' and background_label_subcalss_removed[jj - 1] != 'street': # If not corrected already (it should not be 'stop'), do:
                    background_label_subcalss_removed[jj] = 'street sign'                   
        # *********************************************************************   

        
        # *********************** Correction of 'table' ***********************
        # Specific correction
        L_object_context = len(background_label_subcalss_removed)
        for jj in range(L_object_context):
            name_check = background_label_subcalss_removed[jj]
            if jj == 0 and name_check == 'table': # The first word is 'table'.
                background_label_subcalss_removed[jj] = 'dining table'                 
        # *********************************************************************


        # *********************** Correction of 'plant' ***********************
        # Specific correction
        L_object_context = len(background_label_subcalss_removed)
        for jj in range(L_object_context):
            name_check = background_label_subcalss_removed[jj]
            if jj == 0 and name_check == 'plant': # The first word is 'plant'.
                background_label_subcalss_removed[jj] = 'potted plant'                 
        # *********************************************************************


        # *********************** Correction of 'remote' **********************
        # Specific correction
        L_object_context = len(background_label_subcalss_removed)
        for jj in range(L_object_context):
            name_check = background_label_subcalss_removed[jj]
            if jj == 0 and name_check == 'control': # The first word is 'control'.
                background_label_subcalss_removed[jj] = 'remote'                 
        # *********************************************************************


        # ************************ Correction of 'ball' ***********************
        # Specific correction
        L_object_context = len(background_label_subcalss_removed)
        for jj in range(L_object_context):
            name_check = background_label_subcalss_removed[jj]
            if jj == 0 and name_check == 'ball': # The first word is 'ball'.
                background_label_subcalss_removed[jj] = 'sports ball'                 
        # *********************************************************************


        # ********* Removing specific redundancies from some objects **********
        def remove_phrase_if_object_present(my_text, object_word, remove_phrase): # This function is removing specific phrases.
            my_text = " ".join(my_text)  # Convert list to string.
            if object_word.lower() in my_text.lower():
                my_text = my_text.replace(remove_phrase, "").replace("  ", " ")  # Avoid extra spaces.
                my_text = my_text.split()  # Convert string to list of words.
            return my_text # my_text.strip()
    
        background_label_subcalss_removed = remove_phrase_if_object_present(background_label_subcalss_removed, "handbag", "chain strap") # The phrase "chain strap" is descriptive to "handbag". Thus, we remove it.
        # *********************************************************************                
                
        

        # my_list = background_label_subcalss_removed.split()  # Properly splits into words 
        if isinstance(background_label_subcalss_removed, list):  
            background_label = ' '.join(background_label_subcalss_removed)  # Converting list to str
        elif isinstance(background_label_subcalss_removed, str): 
            background_label = background_label_subcalss_removed
        
        
        
        # *********************************************************************
        # ********************* Object/context separation *********************
        str2word = background_label.split() # Note: .split() breaks the string into words based on spaces.
        N_words = len(str2word) # The number of words
        My_object = '' # Default value for object
        My_context = '' # Default value for context
        # object_label = '' # initialisation
        # context_label = '' # initialisation
        if N_words == 0:
            # We can't communication anything from the scene. In this case, we transmit the default values, which are void.
            print("We can't communication anything from the scene.")
            
        elif N_words == 1:
            My_object = str2word[0]
            
            for ii in range(1, N_objects):
                if My_object == COCO_dataset_category_name_OriginalPaper[ii]:
                    # object_label = ii
                    My_context = default_context[ii - 1]
                    
        elif N_words == 2:
            Noun1 = str2word[0]
            Noun2 = str2word[1]
            
            My_object = Noun1
            My_context = Noun2
            
            # Check for objects containing two words, such as 'eye glasses'.
            All_Nouns = Noun1 + ' ' + Noun2
            for ii in range(1, N_objects + 1): # This loop is from 1 to N_objects, i.e., from 1 to 91 for the COCO dataset.
                if All_Nouns == COCO_dataset_category_name_OriginalPaper[ii]:
                    My_object = All_Nouns
                    # object_label = ii
                    My_context = default_context[ii - 1]
            
        elif N_words == 3:
            Noun1 = str2word[0]
            Noun2 = str2word[1]
            Noun3 = str2word[2]
            
            My_object = Noun1 + ' ' + Noun2 # With the assumption of objects containing two words.
            My_context = Noun3 # With the assumption of a single-word context
            
            # Check for objects containing one word.
            for ii in range(1, N_objects + 1): # This loop is from 1 to N_objects, i.e., from 1 to 91 for the COCO dataset.
                if Noun1 == COCO_dataset_category_name_OriginalPaper[ii]:
                    My_object = Noun1 # Object is single-word.
                    # object_label = ii
                    My_context = Noun2 + ' ' + Noun3 # We force to the context is double-word, due to imperfection.
            
             
        
        print(My_object , My_context)
        # ****************** End of object/context separation *****************
        # *********************************************************************
        
        
        # *********************************************************************
        # *********************** Object-context coding ***********************
        # Note: The following objects list excludes the '__background__' from COCO dataset.
        objects_list = [
            'Person', # 1
            'Bicycle', # 2
            'Car', # 3
            'Motorcycle', # 4
            'Airplane', # 5 
            'Bus', # 6
            'Train', # 7
            'Truck', # 8
            'Boat', # 9
            'Traffic light', # 10
            'Fire hydrant', # 11
            'Street sign', # 12 
            'Stop sign', # 13
            'Parking meter', # 14
            'Bench', # 15
            'Bird', # 16
            'Cat', # 17
            'Dog', # 18
            'Horse', # 19
            'Sheep', # 20
            'Cow', # 21
            'Elephant', # 22
            'Bear', # 23
            'Zebra', # 24
            'Giraffe', # 25
            'Hat', # 26
            'Backpack', # 27
            'Umbrella', # 28
            'Shoe', # 29
            'Eye glasses', # 30
        	'Handbag', # 31
            'Tie', # 32
            'Suitcase', # 33
            'Frisbee', # 34
            'Skis', # 35
            'Snowboard', # 36
            'Sports ball', # 37
            'Kite', # 38
            'Baseball bat', # 39
            'Baseball glove', # 40 
            'Skateboard', # 41 
            'Surfboard', # 42
            'Tennis racket', # 43 
            'Bottle', # 44
            'Plate', # 45
            'Wine glass', # 46
            'Cup', # 47
            'Fork', # 48
            'Knife', # 49
            'Spoon', # 50
            'Bowl', # 51
            'Banana', # 52
            'Apple', # 53
            'Sandwich', # 54
            'Orange', # 55
            'Broccoli', # 56
            'Carrot', # 57
            'Hot dog', # 58
            'Pizza', # 59
            'Donut', # 60
            'Cake', # 61
            'Chair', # 62
            'Couch', # 63
            'Potted plant', # 64
            'Bed', # 65
            'Mirror', # 66
            'Dining table', # 67
            'Window', # 68 
            'Desk', # 69
            'Toilet', # 70
            'Door', # 71
            'TV', # 72
            'Laptop', # 73
            'Mouse', # 74
            'Remote', # 75
            'Keyboard', # 76
            'Cell phone', # 77
            'Microwave', # 78
            'Oven', # 79
            'Toaster', # 80
            'Sink', # 81
            'Refrigerator', # 82
            'Blender', # 83
            'Book', # 84
            'Clock', # 85
            'Vase', # 86
            'Scissors', # 87
            'Teddy bear', # 88
            'Hair drier', # 89
            'Toothbrush', # 90
            'Hair brush' # 91	
            ]
        
        # Note: The following contexts list excludes duplicate contexts that are already in the objects list.
        contexts_list = [
            'Airport', # 1
            'Bakery', # 2
            'Balcony', # 3
            'Bar', # 4
            'Base', # 5
            'Bathroom', # 6
            'Beach', # 7
            'Beans', # 8
            'Bedroom', # 9 
            'Board', # 10
            'Box', # 11
            'Bridge', # 12  
            'Building', # 13
            'Bushes', # 14
            'Cafeteria', # 15 
            'Computer', # 16
            'Cord', # 17
            'Counter', # 18 
            'Court', # 19
            'Deck', # 20
            'Dictionary', # 21 
            'Diningroom', # 22
            'Dirt', # 23
            'Entrance', # 24
            'Face', # 25
            'Farm', # 26
            'Fence', # 27
            'Field', # 28
            'Floor', # 29
            'Flower', # 30
            'Flowers', # 31
            'Footpath', # 32
            'Forest', # 33
            'Fountain', # 34
            'Garden', # 35
            'Grass', # 36
            'Ground', # 37
            'Hand', # 38
            'Highway', # 39 
            'Home', # 40
            'House', # 41
            'Iceberg', # 42
            'Intersection', # 43 
            'Kitchen', # 44
            'Legs', # 45
            'Library', # 46 
            'Livingroom', # 47 
            'Market', # 48
            'Mountain', # 49
            'Napkin', # 50
            'Ocean', # 51
            'Office', # 52
            'Orchard', # 53
            'Pan', # 54
            'Park', # 55
            'Parking', # 56 
            'Pasture', # 57
            'Pavement', # 58
            'Pedestal', # 59
            'Pocket', # 60
            'Pole', # 61
            'Pond', # 62
            'Railway', # 63 
            'Rain', # 64
            'Restaurant', # 65 
            'Restroom', # 66
            'River', # 67
            'Road', # 68
            'Roadside', # 69 
            'Rock', # 70
            'Room', # 71
            'Rug', # 72
            'Savanna', # 73 
            'School', # 74
            'Scooter', # 75
            'Sea', # 76
            'Shelf', # 77
            'Shirt', # 78
            'Shopping', # 79 
            'Sidewalk', # 80
            'Skatepark', # 81
            'Sky', # 82
            'Slope', # 83
            'Snow', # 84
            'Stadium', # 85 
            'Stand', # 86
            'Station', # 87
            'Street', # 88
            'String', # 89
            'Surface', # 90
            'Table', # 91
            'Tray', # 92
            'Tree', # 93
            'Unicorn', # 94 
            'Wall', # 95
            'Wardrobe', # 96 
            'Water', # 97
            'Woods', # 98
            'Workshop', # 99 
            'Yard' # 100
            ]
        
        Dictionary = objects_list + contexts_list # Unified dictionary  
        
        L_Dictionary = len(Dictionary)
        binary_length = len(bin(L_Dictionary)[2:]) 

        My_object_label = '' # initialisation
        My_context_label = '' # initialisation
        
        if len(My_object) != 0: 
            My_object_capitalised = My_object[0].upper() + My_object[1:]
            # Object's label and binary code
            for jj in range(L_Dictionary):
                if My_object_capitalised == Dictionary[jj]:
                    My_object_label = jj # Object's label
                    My_object_binary_code = bin(My_object_label)[2:].zfill(binary_length) # A binary code with the calculated binary length
                    break
        else:
            My_object_capitalised = ''
            My_object_label = L_Dictionary # Object's label
            My_object_binary_code = bin(My_object_label)[2:].zfill(binary_length) # A binary code with the calculated binary length
                               
        
        if len(My_context) != 0:
            My_context_capitalised = My_context[0].upper() + My_context[1:]
            # Context's label and binary code
            for jj in range(L_Dictionary):
                if My_context_capitalised ==  Dictionary[jj]:
                    My_context_label = jj # Context's label
                    My_context_binary_code = bin(My_context_label)[2:].zfill(binary_length) # A binary code with the calculated binary length
                    break
        else:
            My_context_capitalised = ''
            My_context_label = L_Dictionary # Context's label
            My_context_binary_code = bin(My_context_label)[2:].zfill(binary_length) # A binary code with the calculated binary length
            
        # ******************** End of Object-context coding *******************
        # *********************************************************************

        
        Root_Text_plus_category_k = Root_Text + category_k
        if not os.path.exists(Root_Text_plus_category_k):
            os.makedirs(Root_Text_plus_category_k)
        str_output = Root_Text_plus_category_k + '/' + file_name_without_ext + '.txt'
        # Open the file in write mode ('w') and create if it doesn't exist.
        with open(str_output, 'w') as file:
            # Writing words to the file
            # file.write(caption+'\n') # Writing caption
            file.write(My_object_capitalised+'\n') # Writing object followed by a newline character
            file.write(My_context_capitalised) # Writing context (it is written in the next line)
        
        
        # *********************** Visualisation ***********************
        position1 = (20, 50)  # (x, y) coordinates- The first argument is column and the second one is row.
        position2 = (20, 100)  # (x, y) coordinates- The first argument is column and the second one is row.
        position3 = (20, 150)  # (x, y) coordinates- The first argument is column and the second one is row.

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 
        colour = (0, 255, 0)  # Green color in BGR format
        thickness = 1 
        
        # Put text on the image.
        cv2.putText(input_image_BGR, caption, position1, font, font_scale, colour, thickness)
        cv2.putText(input_image_BGR, My_object, position2, font, font_scale, colour, thickness)
        cv2.putText(input_image_BGR, My_context, position3, font, font_scale, colour, thickness)
        
        # Show the image.
        # cv2.imshow("Image with Text", image)
        # cv2.waitKey(0)  # Wait for key press.
        # cv2.destroyAllWindows()
        
        # Save the image.
        filename = category_k # "Output"
        cv2.imwrite(f"{Root_Visualisation}{filename}_{k+1}_{i+1}.jpg", input_image_BGR)
        # cv2.imwrite("output.jpg", input_image_BGR)
        # *************************************************************
        
        
        # *********************************************************************
        # *********************** Semantic segmentation ***********************
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
        
        # Move the input and model to GPU for faster processing if available.
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        
        # Perform inference.
        with torch.no_grad():
            outputs = model(input_batch)
            
        
        boxes = outputs[0]['boxes'] # The boxes is a tensor, where each row belongs to an object. Also, in each row, we have the bounding box information (c_min, r_min, c_max, r_max).   
        labels = outputs[0]['labels'].numpy()
        confidences = outputs[0]['scores'].numpy()
        panoptic_seg = outputs[0]['masks'].numpy()  # Panoptic segmentation output
        
        N_components = len(labels) # The number of recognised components
        binary_image = np.ones((h, w), dtype=np.uint8) * 255 # Initialisation for creating a white image (all pixel values = 255). If failig any segment in the following semantic segmentation process, we return this white image. This "full-segment image" also means that the DSCs model reduces to ordinary raw data communications (i.e., the Shannon model).
        found_match = False # A flag
        if My_object_label != '': 
            mask_area = [0]*N_components # A list of length N_components with all values set to 0
            edge_strength = [0]*N_components # Initialisation
            dist_to_center = [0]*N_components # Initialisation 
            score = [0]*N_components # Initialisation 
            best_score = -np.inf
            best_index = -1
            
            img_center = np.array([w / 2, h / 2])
            max_dist_to_center = np.linalg.norm(img_center)  # Distance from (0, 0) to image center
            total_img_area = h * w 
            
            
            centre_scores = [0]*N_components
            area_scores = [0]*N_components
            edge_scores = [0]*N_components
            
            gray_full = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2GRAY)
            edges_full = cv2.Laplacian(gray_full, cv2.CV_64F)
            total_edge_strength = np.sum(np.abs(edges_full))
            
            for kk in range(N_components):
                if labels[kk] == My_object_label + 1: # This "plus 1" is for matching of definitions.
                    # found_match = True  # Match found
                    c_min, r_min, c_max, r_max = map(int, boxes[kk])
                    out_res = input_image_BGR[r_min:r_max, c_min:c_max, :]
                    if flag_show == 'T':
                        cv2.imshow('foreground object', out_res) 
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    # cv2.imshow('foreground object', out_res) 
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    Segmented = 255*panoptic_seg[kk, 0]
                    thresh = round(confidences[kk]*128) # weighted thresholding
                    
                    mask_area[kk] = np.sum(Segmented > thresh)
                    area_score = mask_area[kk] / total_img_area # Normalisation
                    area_scores[kk] = area_score
                    # area_scores.append(area_score)
                    
                    # print(kk)
                    
                    gray = cv2.cvtColor(out_res, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Laplacian(gray, cv2.CV_64F)
                    edge_strength[kk] = np.mean(np.abs(edges))
                    edge_score = edge_strength[kk] / (total_edge_strength + 1e-8)
                    edge_scores[kk] = edge_score
                    # edge_scores[kk] = edge_strength[kk]
                    # edge_scores.append(edge_strength[kk])
                    
                    # img_center = np.array([w / 2, h / 2])
                    segment_center = np.array([(c_min + c_max) / 2, (r_min + r_max) / 2])
                    dist_to_center[kk] = np.linalg.norm(segment_center - img_center)
                    norm_dist_to_center = dist_to_center[kk] / max_dist_to_center
                    centre_score = 1 - norm_dist_to_center  # Note: closer to centre = higher score
                    centre_scores[kk] = centre_score
                    # centre_scores.append(centre_score)
                    

            # Combine scores (tune weights if needed).
            alpha, beta, gamma = 0.6, 0.7, 0.5 
            # score = alpha * centre_score + beta * area_score + gamma * edge_score
            for kk in range(N_components):
                score = ((alpha/3) * centre_scores[kk] +
                         (beta/3) * area_scores[kk] +
                         (gamma/3) * edge_scores[kk])
                if score > best_score:
                    best_score = score
                    best_index = kk

                    
            # If a matching label was found with best_index:
            if best_index != -1:
                found_match = True
                print("The first If")
                c_min, r_min, c_max, r_max = map(int, boxes[best_index])
                out_res = input_image_BGR[r_min:r_max, c_min:c_max, :]
            
                if flag_show == 'T':
                    cv2.imshow('foreground object', out_res)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
                Segmented = 255 * panoptic_seg[best_index, 0]
                thresh = round(confidences[best_index] * 128)
                binary_image = cv2.threshold(Segmented, thresh, 255, cv2.THRESH_BINARY)[1]
            
                if flag_show == 'T':
                    cv2.imshow('mask', binary_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


        found_match2 = False
        if not found_match and labels.size > 0:
            best_score = -np.inf
            best_index = -1
            img_center = np.array([w / 2, h / 2])
            max_dist_to_center = np.linalg.norm(img_center)  # Distance from (0, 0) to image center
            total_img_area = h * w
            
            centre_scores = [0]*N_components
            area_scores = [0]*N_components
            edge_scores = [0]*N_components
            
            # print("Ali- You found the error!")
            
            gray_full = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2GRAY)
            edges_full = cv2.Laplacian(gray_full, cv2.CV_64F)
            total_edge_strength = np.sum(np.abs(edges_full))

        
            for kk in range(N_components):
                c_min, r_min, c_max, r_max = map(int, boxes[kk])
        
                # Heuristic 1: Distance to image centre
                segment_center = np.array([(c_min + c_max) / 2, (r_min + r_max) / 2])
                dist_to_center = np.linalg.norm(segment_center - img_center)
                norm_dist_to_center = dist_to_center / max_dist_to_center
                centre_score = 1 - norm_dist_to_center  # Note: Closer to centre = higher score
                centre_scores[kk] = centre_score
                # centre_scores.append(centre_score)
        
                # Heuristic 2: Area of the segment
                # area_score = (c_max - c_min) * (r_max - r_min)
                Segmented = 255*panoptic_seg[kk, 0]
                thresh = round(confidences[kk]*128) # weighted thresholding
                area_score = np.sum(Segmented > thresh)
                area_score = area_score / total_img_area # Normalisation
                area_scores[kk] = area_score
                # area_scores.append(area_score)
        
                # Heuristic 3: Edge sharpness
                roi = input_image_BGR[r_min:r_max, c_min:c_max, :]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edge_strength = np.mean(np.abs(edges))
                edge_score = edge_strength / (total_edge_strength + 1e-8)
                edge_scores[kk] = edge_score
                # edge_scores.append(edge_score)

        
            # Combine scores (tune weights if needed)
            alpha, beta, gamma = 0.6, 0.7, 0.5 
            # score = alpha * centre_score + beta * area_score + gamma * edge_score
            for kk in range(N_components):
                score = ((alpha/3) * centre_scores[kk] +
                         (beta/3) * area_scores[kk] +
                         (gamma/3) * edge_scores[kk])
                if score > best_score:
                    best_score = score
                    best_index = kk
        
        
            # Use the best segment found.
            if best_index != -1:
                found_match2 = True
                print("The second If")
                c_min, r_min, c_max, r_max = map(int, boxes[best_index])
                out_res = input_image_BGR[r_min:r_max, c_min:c_max, :]
        
                if flag_show == 'T':
                    cv2.imshow('foreground object', out_res)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
                Segmented = 255 * panoptic_seg[best_index, 0]
                thresh = round(confidences[best_index] * 128)
                binary_image = cv2.threshold(Segmented, thresh, 255, cv2.THRESH_BINARY)[1]
        
                if flag_show == 'T':
                    cv2.imshow('mask', binary_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
        
        
        Root_Segmented_plus_category_k = Root_Segmented + category_k

        str_output = Root_Segmented_plus_category_k + '/' + file_name_without_ext + '.bmp'
        if not os.path.exists(Root_Segmented_plus_category_k):
            os.makedirs(Root_Segmented_plus_category_k)
        cv2. imwrite(str_output, binary_image)
        # ******************** End of semantic segmentation *******************
        # *********************************************************************
        
        
        
        print(k+1, i+1) # Counters of folder and sub-folder
        
        