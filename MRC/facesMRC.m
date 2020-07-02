% This is a MATLAB file that would have the entire MRC algorithm and faces
% code in it.

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




