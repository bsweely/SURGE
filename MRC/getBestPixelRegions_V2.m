function bestPixelRegions = getBestPixelRegions(pixelBoxArrays, Fs, bandpassFilter, error, numBestBoxes, pulseRate)
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
    
    pixelRegions(pixelBox).rois = pixelBoxArrays(1).frame(pixelBox).pixelBoxInstance.roiCoords;
    
    % p = pixelRegions(pixelBox).rIntensities
    
    X = [pixelRegions(pixelBox).rIntensities(1:qtyOfFrames); 
        pixelRegions(pixelBox).gIntensities(1:qtyOfFrames); 
        pixelRegions(pixelBox).bIntensities(1:qtyOfFrames)]; 

    % Best component selection
    [pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal
    ica_s1 = pulse_ica(1,:);
    ica_s2 = pulse_ica(2,:);
    ica_s3 = pulse_ica(3,:);

    % FFT to select which ICA source to use for further filtering 
    N = length(pulse_ica); % same as es 
    freq = 0:Fs/N:Fs/2;

    source1_fft = fft(ica_s1); % fft of each source 
    s1 = abs(source1_fft/N); 
    s1_power_fft = s1(1:N/2+1).^2; 
    s1_power_fft(2:end-1) = 2*s1_power_fft(2:end-1); 

    source2_fft = fft(ica_s2); 
    s2 = abs(source2_fft/N); 
    s2_power_fft = s2(1:(N/2)+1).^2; 
    s2_power_fft(2:end-1) = 2*s2_power_fft(2:end-1); 

    source3_fft = fft(ica_s3); 
    s3 = abs(source3_fft/N); 
    s3_power_fft = s3(1:N/2+1).^2; 
    s3_power_fft(2:end-1) = 2*s3_power_fft(2:end-1); 

    figure(1) % Select the ICA source with the highest peak within the normal HR freq (0.8 to 3 in the freq domain, 48-180 bpm in the time domain)) 
    subplot(3,1,3)
    plot(freq, s3_power_fft)
    subplot(3,1,1)
    plot(freq, s1_power_fft)
    subplot(3,1,2)
    plot(freq, s2_power_fft)

    freq_range = find(freq>=.8 & freq<=3) ; % extracts the indices of the frequencies that are within the range of 0.8 and 3 
    source_select_array = [max(s1_power_fft(freq_range)), max(s2_power_fft(freq_range)), max(s3_power_fft(freq_range))]; % finds the max of each source of the corresponding indices of the frequencies between 0.8 and 3 
    [source, source_number] = max(source_select_array); % finds which source has the max in the freq range compared to the others 

    if source_number == 1 % determines which source will be the 'data' that is process through the rest of the code , should only have to happen once with each moving interval 
        data = ica_s1 ; % can follow the variable name 'data' through processing 
    elseif source_number == 2 
        data = ica_s2 ;
    else
        data = ica_s3 ;
    end
    
    % Moving Average
    data_mov_avg = smoothdata(data, 'movmean', 5); % trial moving avg function 'smoothdata'

    figure (2) % before and after moving average to display smoothing of data 
    plot(1:N, data,'r') 
    hold on 
    plot(1:N, data_mov_avg,'g')

    % Bandpass Filter
    % low = .8/Fs;
    % high = 3/Fs;
    % [b,a]=butter(2,[low,high]);
    % h = fvtool(b,a);
    [data_bp_filter,d]=bandpass(data_mov_avg,[0.8 3],Fs);

    % FFT to find HR Freq 
    data_fft = fft(data_bp_filter); 
    P2 = abs(data_fft/N); 
    data_power_fft = P2(1:N/2+1) ; 
    data_power_fft(2:end-1) = 2*data_power_fft(2:end-1); 

    figure (3) 
    freq = 0:Fs/N:Fs/2;
    plot(freq, data_power_fft,'b') 

    % Taking out peaks outside of 60 and 120 bpm (1 and 2 Hz, respectively)
    data_power_fft = data_power_fft.*(freq>1);
    data_power_fft = data_power_fft.*(freq<2);

    % peak detector
    [maxPeak, index] = maxk(data_power_fft, 5); % getting max peak and index
    Freq_avg = freq(index)
    HR_avg = freq(index) * 60
    
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
    ARange = 10*log10(data_power_fft).*(freq>ALower);
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
    
    BRange = 10*log10(data_power_fft).*(freq>BLower);
    BRange = BRange.*(freq<BUpper); % PSD within bandpass filter
    
    % Getting bandpass integral as described in the Kumar paper
    B = cumtrapz(BDomain, BRange);
    B = abs(B(length(B))); % Getting the integral here
    
    pixelRegions(pixelBox).gmetrics = abs(A./(B - A));
    g = pixelRegions(pixelBox).gmetrics; % The goodness metric calculation for this particular pixelBox
    
    pixelBoxCounter = pixelBoxCounter + 1; % incrementing the pixel box that is being analyzed
    figureCounter = figureCounter + 3;
end

%% returning the top pixel box(es) for tracking PPG 

% Sorting based on the struct itself, not converting it to a table to sort
[x,idx]=sort([pixelRegions.gmetrics], 'descend');
clear x;

bestPixelRegions=pixelRegions(idx);

% Returning as many of the best pixel regions as are available out of the
% requested number. So, if 10 are requested but there are only 8 total,
% then it will return 8. If 20 were available, it would return 10

if length(idx) >= numBestBoxes
    bestPixelRegions = bestPixelRegions(1:numBestBoxes)
% else
%     return bestPixelRegions as is
end

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
    
    
    
    
