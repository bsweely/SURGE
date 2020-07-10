clear; close all; % clc;

Fs = 30; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

%{
mypi=raspi('IP Address','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
%}

end_sample=20; % set how many seconds you want to loop
es = 100;
roi = cell(Fs,1);
numOfInitialFrames = 10; % number of initial frames to acquire before moving average % set to 10 to debug faster
r = zeros(1,numOfInitialFrames + es);
g = zeros(1,numOfInitialFrames + es);
b = zeros(1,numOfInitialFrames + es);

% Importing faceImages mat file to use as standardized data
load('Jeremy_data.mat');

timesPerFrame = zeros(1,numOfInitialFrames + es);
totalTimes = zeros(1,numOfInitialFrames + es);
HR = [];
pixelRegions = [];

% arrays for analzing stats on the most promising regions
modeRIntensities = [];
modeBIntensities = [];
modeGIntensities = [];

% faceImages = dir(fullfile(pwd,'*.jpg'));

% getting initial 200 frames of data
for k = 1:numOfInitialFrames
    tic
    % img = imread(strcat(path,'\',files(k+2).name)); % I have +2 because that is when pictures start
    % img = snapshot(cam);
    img = images(k).snapshot;
    if mod(k,5) == 0 || k == 1
        roi{k} = detectbothcheeks(img);
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
    
    
    yPixelBoxBounds = min(y):20:max(y); % The y coordinates of the 20x20 pixels
    xPixelBoxBounds = min(x):20:max(x); % The x coordinates of the 20x20 pixels
    numOfPixelBoxes = (length(yPixelBoxBounds) - 1)*(length(xPixelBoxBounds) - 1); % number of pixel boxes
    
    % initializing the pixelBoxes array with the pixel boxes
    index = 0;
    rIntensities = zeros(1,numOfPixelBoxes);
    gIntensities = zeros(1,numOfPixelBoxes);
    bIntensities = zeros(1,numOfPixelBoxes);
    % initializing the pixelBoxes array with the pixel boxes
    for col = 1:length(yPixelBoxBounds) - 1
        for row = 1:length(xPixelBoxBounds) - 1
            index = index + 1;
            Red_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,1); 
            Green_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,2); 
            Blue_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,3); 
            rIntensities(index) = sum(sum(Red_ROI)); % intensity -> PPG
            gIntensities(index) = sum(sum(Green_ROI));
            bIntensities(index) = sum(sum(Blue_ROI));
        end
    end

    % collecting the strongest intensites for the R,G,B data
    r(k) = max(rIntensities);
    b(k) = max(bIntensities);
    g(k) = max(gIntensities);
    
    % collecting stats on the regions with most intensity
    modeRIntensities = horzcat(modeRIntensities, find(rIntensities == max(rIntensities)));
    modeBIntensities = horzcat(modeBIntensities, find(bIntensities == max(bIntensities)));
    modeGIntensities = horzcat(modeGIntensities, find(gIntensities == max(gIntensities)));
    
    t = tic;
    timesPerFrame(k) = toc(t);

    if k == 1
        totalTimes(k) = timesPerFrame(k);
    else
        totalTimes(k) = timesPerFrame(k) + totalTimes(k-1);
    end
end

i = numOfInitialFrames; % starting at one index in front of the initial frames for the new data
for numOfIterations = 1:1 
    i = i + 1;
    length_HR = length(HR); % used for debugging
    
    % Getting new PPG data
    for i = i:(i+es-1)
        tic
        % img = imread(strcat(path,'\',files(i+2).name)); % I have +2 because that is when pictures start
        % img = snapshot(cam);
        img = images(i).snapshot;
        if mod(i,5) == 0 || i == 1
            roi{i} = detectbothcheeks(img);
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


        yPixelBoxBounds = min(y):20:max(y); % The y coordinates of the 20x20 pixels
        xPixelBoxBounds = min(x):20:max(x); % The x coordinates of the 20x20 pixels
        numOfPixelBoxes = (length(yPixelBoxBounds) - 1)*(length(xPixelBoxBounds) - 1); % number of pixel boxes
        
        index = 0;
        % initializing the pixelBoxes array with the pixel boxes
        for col = 1:length(yPixelBoxBounds) - 1
            for row = 1:length(xPixelBoxBounds) - 1
                index = index + 1;
                Red_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,1); 
                Green_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,2); 
                Blue_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,3); 
                rIntensities(index) = sum(sum(Red_ROI)); % intensity -> PPG
                gIntensities(index) = sum(sum(Green_ROI));
                bIntensities(index) = sum(sum(Blue_ROI));
            end
        end

        % collecting the strongest intensites for the R,G,B data
        r(i) = max(rIntensities);
        b(i) = max(bIntensities);
        g(i) = max(gIntensities);
        
        % collecting the statistics on which regions of the face are 
        % the most intense to see if there is a trend
        
        modeRIntensities = horzcat(modeRIntensities, find(rIntensities == max(rIntensities)));
        modeBIntensities = horzcat(modeBIntensities, find(bIntensities == max(bIntensities)));
        modeGIntensities = horzcat(modeGIntensities, find(gIntensities == max(gIntensities)));
        
        t = tic;
        timesPerFrame(i) = toc(t);

        if i == 1
            totalTimes(i) = timesPerFrame(i);
        else
            totalTimes(i) = timesPerFrame(i) + totalTimes(i-1);
        end
    end
    
%     % resizing data to have newest 200 frames of data by taking out the
%     % first 20 frames, which reduces these arrays of 220 elements to 200
%     % elements
%     if length(r) > numOfInitialFrames
%         r = reduceToLastNIndices(r, numOfInitialFrames);
%         g = reduceToLastNIndices(g, numOfInitialFrames);
%         b = reduceToLastNIndices(b, numOfInitialFrames);
%         timesPerFrame = reduceToLastNIndices(timesPerFrame, numOfInitialFrames);
%         totalTimes = reduceToLastNIndices(totalTimes, numOfInitialFrames);
%     end
    
    % detrend
    r_detrend = detrend(r);
    b_detrend = detrend(b);
    g_detrend = detrend(g);
    
    % normalize
    r_norm = normalize(r_detrend);
    b_norm = normalize(b_detrend);
    g_norm = normalize(g_detrend);

    % ICA feature selection OR PCA
    % X = [totalTimes; r_norm; b_norm; g_norm];
    % X = [totalTimes; timesPerFrame; r_norm; b_norm; g_norm]; 
    % X = [totalTimes; timesPerFrame; r_detrend; b_detrend; g_detrend]; 
    X = [totalTimes; timesPerFrame; r_norm; b_norm; g_norm]; 
    % [pulse_ica, W, T, mu] = kICA(X,5); % changed to 3 source, find best PPG signal

    % Power Spectral Density to select which component to use
    t = 0:1/Fs:1-1/Fs;
    N = length(X(5,:));
    xdft = fft(X(5,:));
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
    best_comp = 5; % green channel
    Xb = X(5,:); % Xb = X([1 5],:); % took out totalTimes b/c no current face detection
    % Xb = X;

    % Moving Average
    %{
    sw_size = 5; % window size
    pulse_sw = zeros(length(Xb),1);

    for ii = 1:length(psdx)
        if ii > sw_size
              pulse_sw(ii) = mean(Xb(ii-sw_size:ii));
        end
    end
    %}

    % Bandpass Filter
    [filter_out,d]=bandpass(Xb,[0.5 5],Fs);
    Y=filter_out(1:length(Xb));
    pulse_fft = fft(Y) ; 
    P2 = abs(pulse_fft/N); 
    power_fft = P2(1:N/2+1) ; 
    power_fft(2:end-1) = 2*power_fft(2:end-1); 

    figure(2)
    pspectrum(power_fft,Fs) 

    figure (3) 
    plot (freq, 10*log10(power_fft)) 
    
    % figure(4)
    % graph y versus totalTimes for paper

    % peak detector
    % [peaks,locs] = findpeaks(10*log10(power_fft), freq);
    
    [maxPeak, index] = max(10*log10(power_fft)); % getting max peak and index
    HR = horzcat(HR, freq(index)*60);
    % HR = horzcat(HR, locs(1,[1 2 3]) * 60);

    % plot data
    %{
    figure(4)
    plot(1:end_sample, r_sliding_window_avg, 'r');
    hold on
    plot(1:end_sample, g_sliding_window, 'g');
    plot(1:end_sample, b_sliding_window_avg, 'b');
    hold off
    figure(5)
    plot(1:Fs, pulse_ica(1,:), 'r')
    hold on
    plot(1:Fs, pulse_ica(3,:), 'g')
    plot(1:Fs, pulse_ica(2,:), 'b')
    %}
    
    HR
    %%%%%%%%%%%%%%%%%
    
    
    % Loop Frames % if the number of initial frames + es is the limit to
    % the frames that one processes at once, then this if statement will be
    % no longer needed
    if i == numOfInitialFrames + es
        i = numOfInitialFrames;
        % resizing data to have newest 200 frames of data by taking out the
        % first 20 frames, which reduces these arrays of 220 elements to 200
        % elements

        r = reduceToLastNIndices(r, numOfInitialFrames);
        g = reduceToLastNIndices(g, numOfInitialFrames);
        b = reduceToLastNIndices(b, numOfInitialFrames);
        timesPerFrame = reduceToLastNIndices(timesPerFrame, numOfInitialFrames);
        totalTimes = reduceToLastNIndices(totalTimes, numOfInitialFrames);
        
    end
end

modeRIntensities;
modeBIntensities;
modeGIntensities;
    
% Save Data
% save('data_Initials_video#.mat','r','g','b');