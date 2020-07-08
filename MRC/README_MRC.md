## This is an overview of the Maximum Ratio Combining algorithm that is
## used to calculate heart rate (HR) from PPG in faces_MRC.m

# Overview

Maximum Ratio Combining (MRC) is an algorithm that detects a face, divides it up into 20 x 20 size arrays of pixels from the face, and uses ICA to find which 20x20 areas have the greatest intensities, which indicates a high Signal to Noise Ratio (SNR).

# Variables

V(x,y,t) is the intensity of each pixel in an image file.

# Important Considerations

1. Out of RGB, green light is best for using PPG, for the frequencies of the absorbance of oxygenated hemoglobin (Hboxy) and deoxygenated hemoglobin (Hbdeoxy) are within the green light range, to a great degree.

2. However, an experiment with cyan, orange, and green (COG) light gave better accuracies than RGB because the orange and cyan wavelengths better encompass the wavelengths that Hboxy and Hbdeoxy emit.

3. We must programmatically recognize regions that have large light intensity changes, for this implies that there is movement, so these regions should be left out of the ROI.

4. I will not initially program face tracking, for the babies will not be moving their faces a lot in the NICU, so this will reduce computational load considerably.

# Important Findings

1. The MATLAB built-in variable vision.CascadeObjectDetector cannot accurately detect either individual eye or the mouth, for it assumes that all of those orifices are all the right eye, left eye, and mouth. The nose is also not easily recognized.

2. Programming the program to recognize faces every 5 images or so significantly sped up the Programming.

3. We will have to migrate to either Python or C++ because MATLAB cannot run on the Raspberry Pi's processor.

## Steps to Implement MRC algorithm

Studies show that the MRC algorithm is one of the current best algorithms for face detection and PPG calculation. These are the steps that we took to implement the MRC algorithm into our

# Initializing HR detection for average by gathering 200 initial frames

- detect face (vision.CascadeObjectDetector)
- choose largest face in the image (rPPG extractFaceFromVideo.m, detectFace.m)
- get array of intensities for the whole frame in terms of V(x,y,t)
- divide array into 20x20 pixel arrays and put those into another array ()
- bandpass filter between 0.5 and 5 Hz (in old faces code)
- ICA find the best of the 20x20 pixel arrays in the code for feature selection (in RunMe.m in rPPG repo)



```
