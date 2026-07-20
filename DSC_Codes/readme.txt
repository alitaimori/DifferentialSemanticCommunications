
The main implementation codes of the Differential Semantic Communications (DSCs) project for the use case of salient object transmission are as follows.
Please note that we run algorithms as one-shot transmission (shared protocol and/or shared feedback mechanism like source compression strategies) as the proof-of-concept.
However, they can be run between distant Alice and Bob agents that communicate interactively.


1) Transmitter_DSC.py
This Python code gets input images and gives related Textual Information, Mask Information, and Salient Object Pixels. 
It first extracts the Object-Context nouns (using Image Captioning and Caption Distillater), and then performs Semantic Segmentation.


2) Receiver_DSC.py
This code gets the received information of Textual Information, Mask Information, and Salient Object Pixels as inputs and generates an estimation of the source images.
It first generate an Prompt and then applies Diffusion Inpainting.


3) PerfComparison.py
We use the code "PerfComparison.py" to evaluate the results of DSCs.
This code can be also used for PNG, JPEG, and IMAT (see the journal paper of DSCs).
Note 1: While evaluating the competing approaches of PNG, JPEG, and IMAT, make sure that the following block of codes (Lines 263 to 266 in the programme) is uncommented as:
        ID_not[0] = 1000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        ID_not[1] = 2000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        ID_hat[0] = 3000 # This is set for the competing approaches of PNG, JPEG, and IMAT. 
        ID_hat[1] = 4000 # This is set for the competing approaches of PNG, JPEG, and IMAT.

Note 2: The evaluation mechanism for the Language-oriented method is slightly different.
Please refer its original paper for details.  
 

Tips:
I) Make sure you use your own root of files on your machine to be able to run codes (e.g., source images, etc.).
This should be set at the start of programmes.

II) Check loading of required pretrained models on yor machine properly, e.g., StableDiffusionInpaintPipeline.

III) Install any prerequisite libraries that do not exist in your machine.    

IV) The algorithms are processed on CPU. However, the user can expand it for GPU (if available and required for speeding up computations).     

