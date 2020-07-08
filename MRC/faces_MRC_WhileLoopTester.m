clear; close all; clc;

Fs = 30; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);

end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);
numOfInitialFrames = 10; % number of initial frames to acquire before moving average % set to 10 to debug faster
%{
% Test image files
path = strcat("C:\Users\jrsho\Desktop\myFacePictures");
files = dir(fullfile(path,"*"));
length_files = length(files)
%}

timesPerFrame = [];
totalTimes = [];
HR = [];
pixelRegions = [];

% getting initial 200 frames of data
for k = 1:numOfInitialFrames + es
    tic
    % img = imread(strcat(path,'\',files(i+2).name)); % I have +2 because that is when pictures start
    img = snapshot(cam);
    if mod(k,5) == 0 || k == 1
        roi{k} = detectfaces_V2(img);
    else
        roi{k} = roi{k-1}; 
    end 
    if roi{k}==1
        if k>1 
            roi{k} = roi{k-1};
        else 
            roi{k} = [250,150;300,250;250,250;300,150];
        end
    end
    
    % Separating roi into 20x20 pixel points
    x = roi{k}(:,2);
    y = roi{k}(:,1);
    
    xDist = max(x) - min(x);
    yDist = max(y) - min(y);
    
    
    yPixelBoxBounds = min(y):20:max(y) % The y coordinates of the 20x20 pixels
    xPixelBoxBounds = min(x):20:max(x) % The x coordinates of the 20x20 pixels
    numOfPixelBoxes = (length(yPixelBoxBounds) - 1)*(length(xPixelBoxBounds) - 1) % number of pixel boxes
    
    % initializing the pixelBoxes array with the pixel boxes
    for col = 1:length(yPixelBoxBounds) - 1
        for row = 1:length(xPixelBoxBounds) - 1
            
            Red_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,1); 
            Green_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,2); 
            Blue_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,3); 
            r(k) = sum(sum(Red_ROI)); % intensity -> PPG
            g(k) = sum(sum(Green_ROI));
            b(k) = sum(sum(Blue_ROI));
        end
    end
    t = tic;
    timesPerFrame(k) = toc(t);

    if k == 1
        totalTimes(k) = timesPerFrame(k);
    else
        totalTimes(k) = timesPerFrame(k) + totalTimes(k-1);
    end
end

i = numOfInitialFrames; % starting at one index in front of the initial frames for the new data
for g = 1:50
    i = i + 1;
    length_HR = length(HR); % used for debugging
    
    % Getting new PPG data
    for i = i:(i+es-1)
        tic
        % img = imread(strcat(path,'\',files(i+2).name)); % I have +2 because that is when pictures start
        img = snapshot(cam);
        if mod(i,5) == 0 || i == 1
            roi{i} = detectfaces_V2(img);
        else
            roi{i} = roi{i-1}; 
        end 
        if roi{i}==1
            if i>1 
                roi{i} = roi{i-1};
            else 
                roi{i} = [250,150;300,250;250,250;300,150];
            end
        end

        % Separating roi into 20x20 pixel points
        x = roi{i}(:,2);
        y = roi{i}(:,1);

        xDist = max(x) - min(x);
        yDist = max(y) - min(y);


        yPixelBoxBounds = min(y):20:max(y) % The y coordinates of the 20x20 pixels
        xPixelBoxBounds = min(x):20:max(x) % The x coordinates of the 20x20 pixels
        numOfPixelBoxes = (length(yPixelBoxBounds) - 1)*(length(xPixelBoxBounds) - 1) % number of pixel boxes

        % initializing the pixelBoxes array with the pixel boxes
        for col = 1:length(yPixelBoxBounds) - 1
            for row = 1:length(xPixelBoxBounds) - 1

                Red_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,1); 
                Green_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,2); 
                Blue_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,3); 
                r(i) = sum(sum(Red_ROI)); % intensity -> PPG
                g(i) = sum(sum(Green_ROI));
                b(i) = sum(sum(Blue_ROI));
            end
        end
        
        
        
        t = tic;
        timesPerFrame(i) = toc(t);

        if i == 1
            totalTimes(i) = timesPerFrame(i);
        else
            totalTimes(i) = timesPerFrame(i) + totalTimes(i-1);
        end
    end
    
    % resizing data to have newest 200 frames of data by taking out the
    % first 20 frames, which reduces these arrays of 220 elements to 200
    % elements
    if length(r) > numOfInitialFrames
        r = reduceToLastNIndices(r, numOfInitialFrames);
        g = reduceToLastNIndices(g, numOfInitialFrames);
        b = reduceToLastNIndices(b, numOfInitialFrames);
        timesPerFrame = reduceToLastNIndices(timesPerFrame, numOfInitialFrames);
        totalTimes = reduceToLastNIndices(totalTimes, numOfInitialFrames);
    end
    
    % detrend
    r_detrend = detrend(r);
    b_detrend = detrend(b);
    g_detrend = detrend(g);
    
    % normalize
    r_norm = normalize(r_detrend);
    b_norm = normalize(b_detrend);
    g_norm = normalize(g_detrend);

    % ICA feature selection OR PCA
    X = [timesPerFrame; r_norm; b_norm; g_norm]; 
    [pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal

    % Power Spectral Density to select which component to use
    t = 0:1/Fs:1-1/Fs;
    N = length(pulse_ica);
    xdft = fft(pulse_ica);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/N:Fs/2;

    figure(1)
    plot(freq,10*log10(psdx))
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')

    % Best component selection
    best_comp = 3; % green channel
    Xb = X([1 3],:);

    % Moving Average
    sw_size = 5; % window size
    pulse_sw = zeros(length(Xb),1);

    for ii = 1:length(psdx)
        if ii > sw_size
              pulse_sw(ii) = mean(Xb(ii-sw_size:ii));
        end
    end

    % Bandpass Filter
    [filter_out,d]=bandpass(pulse_sw(sw_size+1:end),[0.5 5],Fs);
    y=filter_out(1:length(Xb)-sw_size);
    pulse_fft=abs(fft(y,length(y)));
    figure(2)
    pspectrum(pulse_fft,Fs) 
    
    % checking to see if the totalTimes is in ascending order. If not, skip
    % this iteration and try again
    issortedErrorCounter = 0;
    if issorted(totalTimes) == 0
        issortedErrorCounter = issortedErrorCounter + 1;
        disp('Non-ascending totalTimes array');
        totalTimes
        timesPerFrame
        if i == 40
            i = 0;
        end
        continue
    end

    % peak detector
    [peaks,locs] = findpeaks(10*log10(pulse_fft), totalTimes(1:length(pulse_fft))); % [peaks,locs] = findpeaks(10*log10(pulse_fft), freq); % old code
    
    % getting frequencies that match with the peaks
    [freqPeaks, HRFreqs] = findpeaks(10*log10(psdx), freq);

    % Evaluating the heart rates
    HR = horzcat(HR, 60*HRFreqs);
    
    % resizing HR to current HR frequencies
    HR = reduceToLastNIndices(HR,200);
    averageHR = mean(HR)
    
    %{
    % plot data
    figure(3)
    plot(1:end_sample, r_sliding_window_avg, 'r');
    hold on
    plot(1:end_sample, g_sliding_window, 'g');
    plot(1:end_sample, b_sliding_window_avg, 'b');
    hold off
    figure(4)
    plot(1:Fs, pulse_ica(1,:), 'r')
    hold on
    plot(1:Fs, pulse_ica(3,:), 'g')
    plot(1:Fs, pulse_ica(2,:), 'b')
    % Save Data
    save('data_Initials_video#.mat','r','b','g');
    %}
    
    HR;
    % Calculating how many heart rates are in the normal range in this dataset
    normHRCount = 0;
    for j = 1:length(HR)
        if HR(j) < 100 && HR(j) > 60
            normHRCount = normHRCount + 1;
        end
    end
    normHRPercentage = 100*normHRCount./length(HR)
    figure(3)
    x = 1:length(HR);
    plot(x,HR);
    xlabel('HR Number in list');
    ylabel('HR');
    title('HR values')
    %%%%%%%%%%%%%%%%%
    
    
    % Loop Frames % if the number of initial frames + es is the limit to
    % the frames that one processes at once, then this if statement will be
    % no longer needed
    if i == numOfInitialFrames + es;
        i = numOfInitialFrames;
    end
end
    
% Save Data
% save('data_Initials_video#.mat','r','g','b');