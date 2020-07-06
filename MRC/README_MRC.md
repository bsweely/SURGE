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

# Important Findings

1. The MATLAB built-in variable vision.CascadeObjectDetector cannot accurately detect either individual eye or the mouth, for it assumes that all of those orifices are all the right eye, left eye, and mouth. The nose is also not easily recognized.

2. Programming the program to recognize faces every 5 images or so significantly sped up the Programming.

3. We will have to migrate to either Python or C++ because MATLAB cannot run on the Raspberry Pi's processor.
