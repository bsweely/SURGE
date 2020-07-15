function bestPixelRegions = getBestPixelRegions(pixelBoxArrays, Fs, totalTimes, timesPerFrame, bandpassFilter, error, numBestBoxes, pulseRate)
% This returns the cleanest Pixelboxes according to the goodness metric

numOfEachPixelBox = length(pixelBoxArrays) % this is the number of instances of each pixel box as catalogged in pixelBoxArrays
% pixelBoxArray(3)
numOfDifferentPixelBoxes = length([pixelBoxArrays.pixelBoxArray])

%% Defining the pixelBox struct 
% This holds the data for a set of pixelBoxInstances that correspond to a
% certain pixel region on the cheek or forehead
pixelRegions(numOfDifferentPixelBoxes) = struct();
% pixelRegions.rIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.gIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.bIntensities = zeros(1,numOfEachPixelBox);
% pixelRegions.gmetrics = zeros(1,numOfEachPixelBox); % goodness metric for this pixel box
% pixelRegions.roi = zeros(2,4); % initializing the roi coordinates
% pixelRegions.HR = 0; % this is the heart rate of the highest peak in the PSD vs. freq graph

%% Getting the pixelRegion information

for i = 1:numOfEachPixelBox
    numOfPixelBoxes = length(pixelBoxArrays(i).pixelBoxArray)
    for j = 1:numOfPixelBoxes
        pixelRegions(i).rIntensities(j) = pixelBoxArrays(i,1).pixelBoxArray(j).rIntensity;
        pixelRegions(i).gIntensities(j) = pixelBoxArrays(i,1).pixelBoxArray(j).gIntensity;
        pixelRegions(i).bIntensities(j) = pixelBoxArrays(i,1).pixelBoxArray(j).bIntensity;
    end
end


i
    
    clear i;
numOfDifferentPixelBoxes % here for debugging
for i = 1:numOfDifferentPixelBoxes
    i % i is here for debugging
    pixelRegions(i).rIntensities = normalize(detrend(pixelRegions(i).rIntensities));
    pixelRegions(i).gIntensities = normalize(detrend(pixelRegions(i).gIntensities));
    pixelRegions(i).bIntensities = normalize(detrend(pixelRegions(i).bIntensities));
    
    % debugging with lengths being printed out
    r_norm_length = length(pixelRegions(i).rIntensities)
    timesLength = length(totalTimes)
    totalTimes;
    
    pixelRegions(i).rois = pixelBoxArrays(1).pixelBoxArray(i).roiCoords;
    
    p = pixelRegions(i).rIntensities
    
    X = [totalTimes(1:numOfPixelBoxes); 
        timesPerFrame(1:numOfPixelBoxes); 
        pixelRegions(i).rIntensities(1:numOfPixelBoxes); 
        pixelRegions(i).gIntensities(1:numOfPixelBoxes); 
        pixelRegions(i).bIntensities(1:numOfPixelBoxes)]; 

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
    f = length(freq)

    % Bandpass Filter
    [filter_out,d]=bandpass(X,bandpassFilter,Fs); % [filter_out,d]=bandpass(X,[0.5 5],Fs);
    Y=filter_out(1:length(X));
    pulse_fft = fft(Y) ; 
    P2 = abs(pulse_fft/N); 
    power_fft = P2(1:N/2+1) ; 
    power_fft(2:end-1) = 2*power_fft(2:end-1); 
    
    figure(i)
    plot(freq,10*log10(psdx))
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')
    
    figure(2)
    pspectrum(power_fft,Fs) 

    figure (3) 
    plot (freq, 10*log10(power_fft)) 
    
    % figure(4)
    % graph y versus totalTimes for paper
    
    [maxPeak, index] = max(10*log10(power_fft)); % getting max peak and index
    % [peaks, locs] = findpeaks(10*log10(power_fft), freq); % getting peaks for analysis
    % HR_found = 60*locs
    pixelRegions(i).HR = freq(index)*60
    
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
    ADomain = ADomain.*(freq<AUpper);
    
    % getting PSD values that correspond to the frequency values above
    ARange = 10*log10(power_fft).*(freq>ALower);
    ARange = abs(ARange.*(freq<AUpper));
    
    % Getting the pulse rate integral as mentioned in the literature (Kumar
    % paper)
    A = trapz(ADomain, ARange)
    A = A(length(A))
    
    % Getting bandpass integral values
    BUpper = bandpassFilter(2); % upper limit of bandpass filter
    BLower = bandpassFilter(1); % lower limit of bandpass filter
    
    BDomain = freq.*(freq>BLower)
    BDomain = BDomain.*(freq<BUpper); % freq within bandpass filter
    
    BRange = 10*log10(power_fft).*(freq>BLower);
    BRange = abs(BRange.*(freq<BUpper)); % PSD within bandpass filter
    
    % Getting bandpass integral as described in the Kumar paper
    B = trapz(BDomain, BRange)
    B = B(length(B))
    
    pixelRegions(i).gmetrics = A./(B - A);
    g = pixelRegions(i).gmetrics % The goodness metric calculation for this particular pixelBox
end

%% returning the top pixel box(es) for tracking PPG 

% sorting the pixelRegions by biggest gmetrics in descending order

Table = struct2table(pixelRegions)

sortedTable = sortrows(pixelRegions, 'gmetrics');

pixelRegions = table2struct(sortedTable);

pixelRegions = pixelRegions' % inverting back into original configuration

for iii = 1:numBestBoxes
    bestPixelRegions = [bestPixelRegions pixelRegions(iii)]
end
    
    
    
    
