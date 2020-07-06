% This is a MATLAB file that would have the entire MRC algorithm and faces
% code in it.

% Variables

% V(x,y,t) is the intensity of a certain color (R,G,B,C,O,G) that is
% measures in one frame of a video. (x,y) is a 2D grid of the RGB or COG
% color values in each (x,y) pixel of the image at time t

% Intensity V(x,y,t) can be decomposed into two attributes: the intensity of the
% illumination I(x,y,t) and the reflectance of the surface skin R(x,y,t)

% Intensity of illumination I(x,y,t) refers to the intensity of ambient light
% falling on the face

% Reflectance R(x,y,t) is the light reflected from the skin

% Intensity of illumination is assumed to be constant in the room of
% testing, so changes in I are proportional to changes in R

% Getting the working PPG signals

% code for getting an image
% img = snapshot(cam) bla bla bla from raspi
PPGRegions = TrackPPGRegions(img);

% bandpass filter between [0.5 5] Hz

% reject signals that breach the signal strength threshold Ath = 8
% How to get the A amplitude values in the signals? Is there a function for
% this?

% Camera-based PPG signals are weak, so they are +- 2 value in the 8-bit
% camera. Is this the threshold signal strength Ath = 8?

% Computer a coarse estimate of the pulse rate PRc by doing the following:

% combine the PPG signals from all the remaining ROIs

% Keep track of history of PRc over last 4 epochs, where an epoch is an
% iiteration with one KLT feature set

% if current estimate of PRc is off by +- 24 bpm, then replace current
% estimate with median of the last four estimates

% compute the goodness patric Gi for all remaining ROIs, and ROIs that are
% rejected get Gi = 0.0.

% combine the PPG signals using a weighted average (equation 6)

% compute the Gi for each ROI after each epoch




