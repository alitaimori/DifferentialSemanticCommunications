
# Performance comparison for Differential Semantic Communications (DSCs) as well as competing methods

"""
Ali Taimori
"""

# clear
# reset -f

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # This is to avoid the probable conflicts among packages, libraries, etc.

try:
    from IPython import get_ipython
    get_ipython().magic('clear') # Clearing Console
    get_ipython().magic('reset -f') # Clearing Variable Explorer
except:
    pass

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
# import random
# import os
import math
import statistics
import spacy

import torch
import lpips

import clip
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim



Root_GT_Images = "C:/Users/at3014/OneDrive - Heriot-Watt University/Desktop/HWU Project/DSCs/IEEE TCCN/Codes/DSC/DSC_Dataset/X_resized/*" # Root of Ground Truth Images
Root_GT_Texts = "C:/Users/at3014/OneDrive - Heriot-Watt University/Desktop/HWU Project/DSCs/IEEE TCCN/Codes/DSC/DSC_Dataset/Text_GT/*" # Root of Ground Truth Texts

Root_PR_Images = "C:/Users/at3014/OneDrive - Heriot-Watt University/Desktop/HWU Project/DSCs/IEEE TCCN/Codes/DSC/DSC_Dataset/Res/*" # Root of Predicted Results Images  
Root_PR_Texts = "C:/Users/at3014/OneDrive - Heriot-Watt University/Desktop/HWU Project/DSCs/IEEE TCCN/Codes/DSC/DSC_Dataset/Text/*" # Root of Predicted Results Texts

Root_Segmented_Images = "C:/Users/at3014/OneDrive - Heriot-Watt University/Desktop/HWU Project/DSCs/IEEE TCCN/Codes/DSC/DSC_Dataset/Segmented/*" # Root of Segmented Images

vec_GT_Images = glob.glob(Root_GT_Images)  
N_GT_Images = len(vec_GT_Images) 
vec_GT_Texts = glob.glob(Root_GT_Texts) 

vec_PR_Images = glob.glob(Root_PR_Images)  

vec_PR_Texts = glob.glob(Root_PR_Texts)  

vec_Segmented_Images = glob.glob(Root_Segmented_Images)  

# flag_show = 'F' # A true ('T') or false ('F') flag character for showing information



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


# Note: The following contexts list contains duplicate contexts that are already in the objects list.
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
    'Yard', # 100
    'Plane', # 101
    'Shoe store', # 102
    'Asphalt', # 103
    'Chopping board', # 104
    'Cutting mat', # 105
    'Hotel room', # 106
    'Lamp', # 107
    'Sheet', # 108
    'Gravel', # 109
    'Marble', # 110
    'Broccoli stack', # 111
    'Plate', # 112
    'Moquette', # 113
    'Soil', # 114
    'Leaves', # 115 
    'Store', # 116
    'Shop', # 117
    'Window', # 118
    'Sitting room', # 119
    'Background', # 120
    'Cooker', # 121
    'Jungle', # 122
    'Foliage', # 123
    'Person', # 124
    'Couch', # 125
    'Hall', # 126
    'Tablecloth', # 127
    'Desk', # 128
    'Carpet', # 129
    'Truck', # 130
    'Lake', # 131
    'Oven', # 132
    'Seat', # 133
    'Avenue', # 134
    'Pitch', # 135
    'Tracks', # 136
    'Trees', # 137
    'Bag' # 138
    ]


loss_fn_lpips = lpips.LPIPS(net='alex')  # Note: 'alex' is standard.
loss_fn_lpips.eval()

device = "cpu"  # Since we are not using GPU.

model_clip, preprocess = clip.load("ViT-B/32", device=device)
model_clip.eval()


# nlp = spacy.load("en_core_web_md")  # Load a medium-sized model.
nlp = spacy.load("en_core_web_lg") # Load the large model.

Res_table = [("", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * N_GT_Images
vec_SemMisfit_mean_and_std_PerObject = [(0, 0)] * N_GT_Images
vec_G_id = [] # Initialisation
vec_New_G_id = [] # Initialisation
vec_PSNR = [] # Initialisation
vec_SSIM = [] # Initialisation
vec_Bandwidth = [] # Initialisation
vec_A_object = [] # Initialisation
vec_LPIPS = []  # Initialisation
vec_CLIP = [] # Initialisation
for k in range(N_GT_Images): 
    Root_GT_Images_k = vec_GT_Images[k] + '/*'
    Dir_GT_Images = glob.glob(Root_GT_Images_k) # Directory
    
    Root_GT_Texts_k = vec_GT_Texts[k] + '/*'
    Dir_GT_Texts = glob.glob(Root_GT_Texts_k) # Directory
    
    Root_PR_Images_k = vec_PR_Images[k] + '/*'
    Dir_PR_Images = glob.glob(Root_PR_Images_k) # Directory
    
    Root_PR_Texts_k = vec_PR_Texts[k] + '/*'
    Dir_PR_Texts = glob.glob(Root_PR_Texts_k) # Directory
    
    Root_Segmented_Images_k = vec_Segmented_Images[k] + '/*'
    Dir_Segmented_Images = glob.glob(Root_Segmented_Images_k) # Directory
    
    def Thumbs_removal(Dir):
        Ni = len(Dir) # Updating the number of images for the subject kth after the Thumbs.db removal 
        flag_thumbs = 'F'
        tmp = 0
        Dir_Thumbs_removed = [] # List initialisation
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
    
    Dir_GT_Images = Thumbs_removal(Dir_GT_Images) # Applying the function Thumbs.db removal 
    Ni = len(Dir_GT_Images) # Updating the number of images for the subject kth after the Thumbs.db removal 
    
    Dir_GT_Texts = Thumbs_removal(Dir_GT_Texts) # Applying the function Thumbs.db removal 

    Dir_PR_Images = Thumbs_removal(Dir_PR_Images) # Applying the function Thumbs.db removal 
    
    Dir_PR_Texts = Thumbs_removal(Dir_PR_Texts) # Applying the function Thumbs.db removal 
    
    Dir_Segmented_Images = Thumbs_removal(Dir_Segmented_Images) # Applying the function Thumbs.db removal 
    

    vec_SemanticMisfitPercent_PerImagesOfAnObject = [0.0] * Ni
    vec_NewSemanticMisfitPercent_PerImagesOfAnObject = [0.0] * Ni
    vec_PSNR_PerImagesOfAnObject = [0.0] * Ni
    vec_SSIM_PerImagesOfAnObject = [0.0] * Ni
    vec_Bandwidth_PerImagesOfAnObject = [0.0] * Ni
    vec_A_object_PerImagesOfAnObject = [0.0] * Ni
    vec_LPIPS_PerImagesOfAnObject = [0.0] * Ni
    vec_CLIP_PerImagesOfAnObject = [0.0] * Ni
    for i in range(Ni): # range(Ni)
    
        with open(Dir_GT_Texts[i], 'r') as file:
            lines_not = file.readlines()
            row1_not = lines_not[0].strip()
            row2_not = lines_not[1].strip()
            
        
        with open(Dir_PR_Texts[i], 'r') as file:
            lines_hat = file.readlines()
            row1_hat = lines_hat[0].strip() if len(lines_hat) > 0 else ''
            row2_hat = lines_hat[1].strip() if len(lines_hat) > 1 else ''

        path = os.path.dirname(Dir_GT_Images[i])
        category_k = os.path.basename(path)
        
        filename = Dir_GT_Images[i]
        filename = os.path.normpath(filename)
        filename = filename.replace("\\", "/")
        image_name = os.path.basename(filename)
        file_name_without_ext = os.path.splitext(image_name)[0]
        
        if category_k != 'TV': 
            label = objects_list.index(category_k.capitalize()) + 1
        else:
            label = objects_list.index(category_k) + 1
        
        
        # ******************** Calculations for comparision *******************        
        X_not = cv2.imread(Dir_GT_Images[i], cv2.IMREAD_GRAYSCALE)

        X_hat = cv2.imread(Dir_PR_Images[i], cv2.IMREAD_GRAYSCALE)
        
        h, w = X_not.shape[:2] # The size of the image X_not
        L_d = h*w # The length of data, which is the number of pixels here.

        L_i = 2 # The length of Information

        L_id = L_i + L_d # The sum of the length of Information plus Data

        ID_not = np.zeros(L_id) # Initialisation for the ground truth Information plus Data (ID)
        
        if row1_not in objects_list:
            ID_not[0] = objects_list.index(row1_not)
        else:
            ID_not[0] = len(objects_list)
        
        
        if row2_not in contexts_list:
            ID_not[1] = contexts_list.index(row2_not) 
        else:
            ID_not[1] = len(contexts_list)
        
        

        X_not_vec = X_not.flatten() # Vectorisation
        X_not_vec = X_not_vec.astype(np.float64)
        X_not_vec = X_not_vec / 255.0

        # L_ID = len(ID_not)
        ID_not[2:L_id] = X_not_vec

        ID_hat = np.zeros(L_id) # Initialisation for the predicted Information plus Data (ID) 
        
        if row1_hat in objects_list:
            ID_hat[0] = objects_list.index(row1_hat)
        else:
            ID_hat[0] = len(objects_list) # A number out of range to prevent the Python error "ValueError"
                
        
        if row2_hat in contexts_list:
            ID_hat[1] = contexts_list.index(row2_hat) 
        else:
            ID_hat[1] = len(contexts_list)
            
        
        # ******** Use this block only for the competing methods below. *******
        # ID_not[0] = 1000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        # ID_not[1] = 2000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        # ID_hat[0] = 3000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        # ID_hat[1] = 4000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        # *********************************************************************
        

        X_hat_vec = X_hat.flatten() # Vectorisation
        X_hat_vec = X_hat_vec.astype(np.float64)
        X_hat_vec = X_hat_vec / 255.0
        ID_hat[2:L_id] = X_hat_vec

            
        S_i_numerator = sum(np.sign(abs(ID_hat[0:L_i] - ID_not[0:L_i])))
            

        S_d_numerator = sum(np.abs(ID_hat[2:L_id] - ID_not[2:L_id]))

        S_d_denominator = sum(np.maximum(ID_hat[2:L_id], ID_not[2:L_id]))
            

        # Create a NumPy array.
        # x = np.array([0, 1])
        Nl = 2; # The total number of levels in the hierarchy
        t = np.arange(Nl)

        # Calculate the exponential of each element in the array.
        tau = math.pow(10, -1.06718477) 
        w_NonNormalised = np.exp(-t / tau)
        w_Normalised = w_NonNormalised / sum(w_NonNormalised)



        print(w_Normalised)  



        G_id = 100*(w_Normalised[0]*S_i_numerator + w_Normalised[1]*S_d_numerator)/(w_Normalised[0]*L_i + w_Normalised[1]*S_d_denominator)
        
        
        vec_SemanticMisfitPercent_PerImagesOfAnObject[i] = G_id
        vec_G_id.append(G_id)
        
        
        
        similarity = np.zeros(L_i)
        vec_ones = np.ones(L_i)

        
        word1 = nlp(row1_not)
        word2 = nlp(row1_hat)
        similarity[0] = word1.similarity(word2)
        
        word1 = nlp(row2_not)
        word2 = nlp(row2_hat)
        similarity[1] = word1.similarity(word2)

        # S_i_numerator = sum(np.sign(abs(ID_hat[0:L_i] - ID_not[0:L_i])))
        S_i_numerator = sum(vec_ones[0:L_i] - similarity[0:L_i])
        

        G_id = 100*(w_Normalised[0]*S_i_numerator + w_Normalised[1]*S_d_numerator)/(w_Normalised[0]*L_i + w_Normalised[1]*S_d_denominator)
        
        vec_NewSemanticMisfitPercent_PerImagesOfAnObject[i] = G_id
        vec_New_G_id.append(G_id)
        
        
        PSNR_val = peak_signal_noise_ratio(X_not, X_hat, data_range=255)
        vec_PSNR_PerImagesOfAnObject[i] = PSNR_val
        vec_PSNR.append(PSNR_val)
        
        
        SSIM_index = ssim(X_not, X_hat, data_range=255)
        vec_SSIM_PerImagesOfAnObject[i] = SSIM_index
        vec_SSIM.append(SSIM_index)
        
        # LPIPS
        # Read images in RGB for LPIPS.
        X_not_rgb = cv2.imread(Dir_GT_Images[i])
        X_hat_rgb = cv2.imread(Dir_PR_Images[i])
        
        # Convert BGR (OpenCV) to RGB.
        X_not_rgb = cv2.cvtColor(X_not_rgb, cv2.COLOR_BGR2RGB)
        X_hat_rgb = cv2.cvtColor(X_hat_rgb, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalise to [-1, 1].
        X_not_rgb = X_not_rgb.astype(np.float32) / 255.0
        X_hat_rgb = X_hat_rgb.astype(np.float32) / 255.0
        
        X_not_rgb = (X_not_rgb - 0.5) * 2.0
        X_hat_rgb = (X_hat_rgb - 0.5) * 2.0
        
        # Convert to tensor and reshape.
        tensor_not = torch.from_numpy(X_not_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float()
        tensor_hat = torch.from_numpy(X_hat_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float()
        
        # Compute LPIPS.
        with torch.no_grad():
            lpips_val = loss_fn_lpips(tensor_not, tensor_hat).item()
        
        vec_LPIPS_PerImagesOfAnObject[i] = lpips_val
        vec_LPIPS.append(lpips_val)
        
        
        # ================= CLIP SCORE =================
        
        image = Image.open(Dir_PR_Images[i]).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Recommended: Use GT semantics (better for our research)
        text_prompt = f"A photo of a {row1_not} in a {row2_not}"
        
        text_tokens = clip.tokenize([text_prompt]).to(device)
        
        with torch.no_grad():
            image_features = model_clip.encode_image(image_input)
            text_features = model_clip.encode_text(text_tokens)
        
            # Normalise.
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
            clip_score = (image_features @ text_features.T).item()
        
        vec_CLIP_PerImagesOfAnObject[i] = clip_score
        vec_CLIP.append(clip_score)
        
        
        # Bandwidth calculation
        Segmented_Image = cv2.imread(Dir_Segmented_Images[i], cv2.IMREAD_GRAYSCALE) # Read the segmented image in grayscale mode.
        _, Segmented_Image = cv2.threshold(Segmented_Image, 127, 255, cv2.THRESH_BINARY) # Apply binary thresholding. Although the segmented image is already in binary format (e.g., 1-bit BMP), OpenCV still reads it as 8-bit grayscale. Thus, we will need to threshold it again to be sure it is binary.

        Segmented_Image = Segmented_Image // 255  # Convert from 0/255 to 0/1.        
        N_pixels_SalientObject = np.count_nonzero(Segmented_Image) # Count foreground pixels (the pixels with value 1).
        
        N_ColourChannels = 3
        Bytes_SalientObjectColours = N_pixels_SalientObject * N_ColourChannels
        Bytes_object = 1 # One byte to describe the object (e.g., object label or ID)
        Bytes_context = 1 # One byte for context information
        Bytes_TextualInformation = Bytes_object + Bytes_context
        Bytes_BinaryMask = L_d/8 # One bit per pixel- Reporting in terms of byte (B)
        Bytes_feedback = 0.25 # one-fourth byte (which is equal to two bits) 
        Bandwidth = Bytes_TextualInformation + Bytes_BinaryMask + Bytes_SalientObjectColours + Bytes_feedback # Total bandwidth in terms of byte (B) 
        Bandwidth = Bandwidth/1000 # Convert bandwidth to Kilobyte (KB) for reporting file sizes in SI units. Note: Division by 1024 can be used in binary units. 
        vec_Bandwidth_PerImagesOfAnObject[i] = Bandwidth
        vec_Bandwidth.append(Bandwidth)
        
        
        A_object = 100*N_pixels_SalientObject/L_d # Object area
        vec_A_object_PerImagesOfAnObject[i] = A_object
        vec_A_object.append(A_object)
        
        
        print(G_id)
        # **************** End of calculations for comparision ****************

        
        
        
        print(k+1, i+1) # Counters of folder and sub-folder
        
    mean_SemMisfit_PerImagesOfAnObject = statistics.mean(vec_SemanticMisfitPercent_PerImagesOfAnObject) # Sample mean
    mean_SemMisfit_PerImagesOfAnObject = round(mean_SemMisfit_PerImagesOfAnObject, 2)
    if Ni>1:
        std_SemMisfit_PerImagesOfAnObject = statistics.stdev(vec_SemanticMisfitPercent_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_SemMisfit_PerImagesOfAnObject = 0.0
    
    std_SemMisfit_PerImagesOfAnObject = round(std_SemMisfit_PerImagesOfAnObject, 2)
    
    
    mean_NewSemMisfit_PerImagesOfAnObject = statistics.mean(vec_NewSemanticMisfitPercent_PerImagesOfAnObject) # Sample mean
    mean_NewSemMisfit_PerImagesOfAnObject = round(mean_NewSemMisfit_PerImagesOfAnObject, 2)
    if Ni>1:
        std_NewSemMisfit_PerImagesOfAnObject = statistics.stdev(vec_NewSemanticMisfitPercent_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_NewSemMisfit_PerImagesOfAnObject = 0.0
    
    std_NewSemMisfit_PerImagesOfAnObject = round(std_NewSemMisfit_PerImagesOfAnObject, 2)
        
    
    mean_PSNR_PerImagesOfAnObject = statistics.mean(vec_PSNR_PerImagesOfAnObject) # Sample mean
    mean_PSNR_PerImagesOfAnObject = round(mean_PSNR_PerImagesOfAnObject, 2)
    if Ni>1:
        std_PSNR_PerImagesOfAnObject = statistics.stdev(vec_PSNR_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_PSNR_PerImagesOfAnObject = 0.0
    
    std_PSNR_PerImagesOfAnObject = round(std_PSNR_PerImagesOfAnObject, 2)
    
    
    mean_SSIM_PerImagesOfAnObject = statistics.mean(vec_SSIM_PerImagesOfAnObject) # Sample mean
    mean_SSIM_PerImagesOfAnObject = round(mean_SSIM_PerImagesOfAnObject, 2)
    if Ni>1:
        std_SSIM_PerImagesOfAnObject = statistics.stdev(vec_SSIM_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_SSIM_PerImagesOfAnObject = 0.0    
    
    std_SSIM_PerImagesOfAnObject = round(std_SSIM_PerImagesOfAnObject, 2)
    
    mean_LPIPS_PerImagesOfAnObject = statistics.mean(vec_LPIPS_PerImagesOfAnObject)
    mean_LPIPS_PerImagesOfAnObject = round(mean_LPIPS_PerImagesOfAnObject, 4)
    
    if Ni > 1:
        std_LPIPS_PerImagesOfAnObject = statistics.stdev(vec_LPIPS_PerImagesOfAnObject)
    else:
        std_LPIPS_PerImagesOfAnObject = 0.0
    
    std_LPIPS_PerImagesOfAnObject = round(std_LPIPS_PerImagesOfAnObject, 4)
    
    mean_CLIP_PerImagesOfAnObject = round(statistics.mean(vec_CLIP_PerImagesOfAnObject), 4)

    if Ni > 1:
        std_CLIP_PerImagesOfAnObject = round(statistics.stdev(vec_CLIP_PerImagesOfAnObject), 4)
    else:
        std_CLIP_PerImagesOfAnObject = 0.0
    
    
    mean_Bandwidth_PerImagesOfAnObject = statistics.mean(vec_Bandwidth_PerImagesOfAnObject) # Sample mean
    mean_Bandwidth_PerImagesOfAnObject = round(mean_Bandwidth_PerImagesOfAnObject, 2)
    if Ni>1:
        std_Bandwidth_PerImagesOfAnObject = statistics.stdev(vec_Bandwidth_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_Bandwidth_PerImagesOfAnObject = 0.0    
    
    std_Bandwidth_PerImagesOfAnObject = round(std_Bandwidth_PerImagesOfAnObject, 2)
    
    
    mean_A_object_PerImagesOfAnObject = statistics.mean(vec_A_object_PerImagesOfAnObject) # Sample mean
    mean_A_object_PerImagesOfAnObject = round(mean_A_object_PerImagesOfAnObject, 2)
    if Ni>1:
        std_A_object_PerImagesOfAnObject = statistics.stdev(vec_A_object_PerImagesOfAnObject)  # Sample standard deviation  
    else:
        std_A_object_PerImagesOfAnObject = 0.0    
    
    std_A_object_PerImagesOfAnObject = round(std_A_object_PerImagesOfAnObject, 2)
    
    
    vec_SemMisfit_mean_and_std_PerObject[k] = (mean_SemMisfit_PerImagesOfAnObject, std_SemMisfit_PerImagesOfAnObject)
    
    Res_table[k] = (
    category_k, label, Ni,
    mean_A_object_PerImagesOfAnObject, std_A_object_PerImagesOfAnObject,
    mean_Bandwidth_PerImagesOfAnObject, std_Bandwidth_PerImagesOfAnObject,
    mean_PSNR_PerImagesOfAnObject, std_PSNR_PerImagesOfAnObject,
    mean_SSIM_PerImagesOfAnObject, std_SSIM_PerImagesOfAnObject,
    mean_LPIPS_PerImagesOfAnObject, std_LPIPS_PerImagesOfAnObject,
    mean_CLIP_PerImagesOfAnObject, std_CLIP_PerImagesOfAnObject,
    mean_SemMisfit_PerImagesOfAnObject, std_SemMisfit_PerImagesOfAnObject,
    mean_NewSemMisfit_PerImagesOfAnObject, std_NewSemMisfit_PerImagesOfAnObject
)

Res_table_sorted = sorted(Res_table, key=lambda x: x[1]) # Sorting the list of tuples in ascending order


mean_SemMisfit = statistics.mean(vec_G_id) # Sample mean
std_SemMisfit = statistics.stdev(vec_G_id)  # Sample standard deviation 

mean_NewSemMisfit = statistics.mean(vec_New_G_id) # Sample mean
std_NewSemMisfit = statistics.stdev(vec_New_G_id)  # Sample standard deviation   

mean_PSNR = statistics.mean(vec_PSNR) # Sample mean
std_PSNR = statistics.stdev(vec_PSNR)  # Sample standard deviation  

mean_SSIM = statistics.mean(vec_SSIM) # Sampl]e mean
std_SSIM = statistics.stdev(vec_SSIM)  # Sample standard deviation  

mean_LPIPS = statistics.mean(vec_LPIPS)
std_LPIPS = statistics.stdev(vec_LPIPS)

mean_CLIP = statistics.mean(vec_CLIP)
std_CLIP = statistics.stdev(vec_CLIP)

mean_Bandwidth = statistics.mean(vec_Bandwidth) # Sample mean
std_Bandwidth = statistics.stdev(vec_Bandwidth)  # Sample standard deviation 

mean_ObjectArea = statistics.mean(vec_A_object) 
std_ObjectArea = statistics.stdev(vec_A_object)

      

# FID
import shutil

FID_GT_path = "FID_GT"
FID_PR_path = "FID_PR"

os.makedirs(FID_GT_path, exist_ok=True)
os.makedirs(FID_PR_path, exist_ok=True)

counter = 0

for k in range(N_GT_Images):
    Root_GT_Images_k = vec_GT_Images[k] + '/*'
    Dir_GT_Images = glob.glob(Root_GT_Images_k)
    Dir_GT_Images = Thumbs_removal(Dir_GT_Images)
    
    Root_PR_Images_k = vec_PR_Images[k] + '/*'
    Dir_PR_Images = glob.glob(Root_PR_Images_k)
    Dir_PR_Images = Thumbs_removal(Dir_PR_Images)
    
    for i in range(len(Dir_GT_Images)):
        shutil.copy(Dir_GT_Images[i], f"{FID_GT_path}/{counter}.png")
        shutil.copy(Dir_PR_Images[i], f"{FID_PR_path}/{counter}.png")
        counter += 1



import subprocess
import sys

try:
    result = subprocess.check_output(
        [sys.executable, "-m", "pytorch_fid", "FID_GT", "FID_PR"],
        stderr=subprocess.STDOUT
    )
    output = result.decode("utf-8")
    print(output)

    FID_value = float(output.split()[-1])

except subprocess.CalledProcessError as e:
    print("FID ERROR:\n")
    print(e.output.decode("utf-8"))
    
    