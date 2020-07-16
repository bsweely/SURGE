function bestPixelRegions = getBestPixelRegions(pixelBoxArrays, Fs, totalTimes, timesPerFrame, bandpassFilter, error, numBestBoxes, pulseRate)
% This returns the cleanest Pixelboxes according to the goodness metric

qtyOfFrames = length(pixelBoxArrays); % this is the number of instances of each pixel box as catalogged in pixelBoxArrays
% frame(3)

% Here, we are getting the number of pixel boxes that were calculated in each frame.
% With face tracking implemented in the future, the number of pixel boxes
% for each frame should be the same. As of not, it is not known whether the
% pixel boxes between each frame correspond to each other. With minimal
% movement of the face, they might correspond enough to reveal an accurate
% reading. 
qtyOfDifferentPixelBoxes = zeros(1,qtyOfFrames);
for frameCount = 1:qtyOfFrames
    qtyOfDifferentPixelBoxes(frameCount) = length(pixelBoxArrays(frameCount).frame);
end

qtyOfDifferentPixelBoxes; % Here for debugging

%% Defining the pixelBox struct 
% This holds the data for a set of pixelBoxInstances that correspond to a
% certain pixel region on the cheek or forehead
pixelRegions(qtyOfDifferentPixelBoxes) = struct();
% pixelRegions.rIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.gIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.bIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.gmetrics = zeros(1,numOfEachPixelBox); % goodness metric for this pixel box
% pixelRegions.roi = zeros(2,4); % initializing the roi coordinates
% pixelRegions.HR = 0; % this is the heart rate of the highest peak in the PSD vs. freq graph

%% Getting the pixelRegion information

% stuff that is printing for debugging
tt = pixelBoxArrays;
t = struct2table(pixelBoxArrays);
e = struct2table(pixelBoxArrays(1,1));
f = struct2table(pixelBoxArrays(1,1).frame);
g = struct2table(pixelBoxArrays(1,1).frame(1));

for i = 1:qtyOfFrames % For each image that we have collected
    for j = 1:qtyOfDifferentPixelBoxes(i)
        i;
        j;
        pixelRegions(j).rIntensities(i) = pixelBoxArrays(i).frame(j).pixelBoxInstance.rIntensity;
        pixelRegions(j).gIntensities(i) = pixelBoxArrays(i).frame(j).pixelBoxInstance.gIntensity;
        pixelRegions(j).bIntensities(i) = pixelBoxArrays(i).frame(j).pixelBoxInstance.bIntensity;
    end
end

pixelBoxCounter = 1; % to help increment through the pixel boxes without indexing out of bounds
figureCounter = 1; % to count the figures that are being produced here
for pixelBox = 1:qtyOfDifferentPixelBoxes(pixelBoxCounter)
    pixelRegions(pixelBox).rIntensities = normalize(detrend(pixelRegions(pixelBox).rIntensities));
    pixelRegions(pixelBox).gIntensities = normalize(detrend(pixelRegions(pixelBox).gIntensities));
    pixelRegions(pixelBox).bIntensities = normalize(detrend(pixelRegions(pixelBox).bIntensities));
    
    % debugging with lengths being printed out
    % r_norm_length = length(pixelRegions(pixelBox).rIntensities)
    timesLength = length(totalTimes);
    totalTimes;
    
    pixelRegions(pixelBox).rois = pixelBoxArrays(2).frame(pixelBox).pixelBoxInstance.roiCoords;
    
    % p = pixelRegions(pixelBox).rIntensities
    
    X = [totalTimes(1:qtyOfFrames); 
        timesPerFrame(1:qtyOfFrames); 
        pixelRegions(pixelBox).rIntensities(1:qtyOfFrames); 
        pixelRegions(pixelBox).gIntensities(1:qtyOfFrames); 
        pixelRegions(pixelBox).bIntensities(1:qtyOfFrames)]; 

    % Power Spectral Density to select which component to use
    t = 0:1/Fs:1-1/Fs;
    N = length(X(5,:));
    % N = N.*10000; % Trying to get more points for frequency to use for
    % goodness metric
    xdft = fft(X(5,:));
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/N:Fs/2;
    f = length(freq);

    % Bandpass Filter
    [filter_out,d]=bandpass(X,bandpassFilter,Fs); % [filter_out,d]=bandpass(X,[0.5 5],Fs);
    Y=filter_out(1:length(X));
    pulse_fft = fft(Y) ; 
    P2 = abs(pulse_fft/N); 
    power_fft = P2(1:N/2+1) ; 
    power_fft(2:end-1) = 2*power_fft(2:end-1); 
    
    %{
    figure(figureCounter)
    plot(freq,10*log10(psdx))
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')
    
    figure(figureCounter+1)
    pspectrum(power_fft,Fs) 

    figure (figureCounter+2) 
    plot (freq, 10*log10(power_fft)) 
    
    % figure(4)
    % graph y versus totalTimes for paper
    %}
    
    [maxPeak, index] = max(10*log10(power_fft)); % getting max peak and index
    % [peaks, locs] = findpeaks(10*log10(power_fft), freq); % getting peaks for analysis
    % HR_found = 60*locs
    pixelRegions(pixelBox).HR = freq(index)*60;
    
    %% Calculating the Goodness Metric

    %{
    The Goodness Metric is calculated as:

    G = A/(B - A), where A = area under the peak of the probable pulse rate in
    the Power Spectral Density (PSD) graph, and B = the area under the PSD
    graph between the lower bound of the bandpass filter and the upper bound of
    the bandpass filter. 

    This essentially calculates how little noise there is surrounding the peak
    of the assumed pulse rate. The higher the G, the cleaner the PPG signal at
    the pulse rate peak. The lower the G, the more noisey the peak pulse rate
    is.
    %}
    
    AUpper = (pulseRate + error)./60; % to get the frequency from pulse rate
    ALower = (pulseRate - error)./60;
    
    % Getting frequencies between the ALower limit and AUpper limit
    ADomain = freq.*(freq>ALower);
    % a1 = freq>ALower;
    ADomain = ADomain.*(freq<AUpper);
    % a2 = freq<AUpper;
    
    % getting PSD values that correspond to the frequency values above
    ARange = 10*log10(power_fft).*(freq>ALower);
    ARange = ARange.*(freq<AUpper);
    
    % Getting the pulse rate integral as mentioned in the literature (Kumar
    % paper)
    A = cumtrapz(ADomain, ARange);
    A = A(length(A)); % Getting the integral here
    
    % Getting bandpass integral values
    BUpper = bandpassFilter(2); % upper limit of bandpass filter
    BLower = bandpassFilter(1); % lower limit of bandpass filter
    
    BDomain = freq.*(freq>BLower);
    % b1 = freq>BLower
    BDomain = BDomain.*(freq<BUpper); % freq within bandpass filter
    % b2 = freq<BUpper
    
    BRange = 10*log10(power_fft).*(freq>BLower);
    BRange = BRange.*(freq<BUpper); % PSD within bandpass filter
    
    % Getting bandpass integral as described in the Kumar paper
    B = cumtrapz(BDomain, BRange);
    B = abs(B(length(B))); % Getting the integral here
    
    pixelRegions(pixelBox).gmetrics = abs(A./(B - A));
    g = pixelRegions(pixelBox).gmetrics % The goodness metric calculation for this particular pixelBox
    
    pixelBoxCounter = pixelBoxCounter + 1; % incrementing the pixel box that is being analyzed
    figureCounter = figureCounter + 3;
end

%% returning the top pixel box(es) for tracking PPG 

% Sorting based on the struct itself, not converting it to a table to sort
[x,idx]=sort([pixelRegions.gmetrics], 'descend')
clear x;
bestPixelRegions=pixelRegions(idx)
bestPixelRegions = bestPixelRegions(1:numBestBoxes)

% sorting the pixelRegions by biggest gmetrics in descending order
%{
Table = struct2table(pixelRegions)

sortedTable = sortrows(pixelRegions, 6, 'descend') % , 'MissingPlacement', 'last');

pixelRegions = table2struct(sortedTable);

pixelRegions = pixelRegions' % inverting back into original configuration
%}
%{
for bestBoxCounter = 1:numBestBoxes
    bestPixelRegions = [bestPixelRegions pixelRegions(bestBoxCounter)]
end
%}
    
    
    
    
