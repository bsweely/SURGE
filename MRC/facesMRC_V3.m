clear; close all; % clc;

Fs = 30; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

mypi=raspi('IP_ADDRESS','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);


end_sample=20; % set how many seconds you want to loop
es = 100;
roiForehead = cell(Fs,1);
roiLeftCheek = cell(Fs,1);
roiRightCheek = cell(Fs,1);

numOfInitialFrames = 10; % number of initial frames to acquire before moving average % set to 10 to debug faster
r = zeros(1,numOfInitialFrames + es);
g = zeros(1,numOfInitialFrames + es);
b = zeros(1,numOfInitialFrames + es);

timesPerFrame = zeros(1,numOfInitialFrames + es);
totalTimes = zeros(1,numOfInitialFrames + es);
HR = [];

% don't include these statements so that we can use these vars as structs,
% not as arrays
% foreheadPixelBoxArray(numOfInitialFrames,1) = struct();
% leftCheekPixelBoxArray = zeros(numOfInitialFrames, 1);
% rightCheekPixelBoxArray = zeros(numOfInitialFrames, 1);

% getting initial 200 frames of data
for k = 1:numOfInitialFrames
    tic
    img = snapshot(cam);
    if mod(k,5) == 0 || k == 1
        [roiForehead{k}, roiLeftCheek{k}, roiRightCheek{k}] = detectCheeksAndForehead_V2(img);
    else
        roiForehead{k} = roiForehead{k-1};
        roiLeftCheek{k} = roiLeftCheek{k-1};
        roiRightCheek{k} = roiRightCheek{k-1};
    end 
    if roiForehead{k}==1
        if k>1 
            roiForehead{k} = roiForehead{k-1};
            roiLeftCheek{k} = roiLeftCheek{k-1};
            roiRightCheek{k} = roiRightCheek{k-1};
        else 
            faceRoi = [250,150;300,250;250,250;300,150];
            x = faceRoi(:,1);
            y = faceRoi(:,2);
            faceWidth = max(x) - min(x);
            faceHeight = max(y) - min(y);
            
            roiForehead{k} = [250,min(y) + 0.5*faceHeight;300,min(y) + 0.75*faceHeight;250,min(y);300,min(y)];
            
            leftCheekX = [min(x); min(x); min(x) + faceWidth*0.4; min(x) + faceWidth*0.4];
            leftCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
            roiLeftCheek{k} = [leftCheekX leftCheekY];

            rightCheekX = [min(x) + faceWidth*0.6; min(x) + faceWidth*0.6; max(x); max(x)];
            rightCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
            roiRightCheek{k} = [rightCheekX rightCheekY];
        end
    end
    
    foreheadPixelBoxArrays(k,1).pixelBoxArray = roiToPixelBoxArray(roiForehead{k}, img);
    leftCheekPixelBoxArrays(k,1).pixelBoxArray = roiToPixelBoxArray(roiLeftCheek{k}, img);
    rightCheekPixelBoxArrays(k,1).pixelBoxArray = roiToPixelBoxArray(roiRightCheek{k}, img);
    
    t = tic;
    timesPerFrame(k) = toc(t);

    if k == 1
        totalTimes(k) = timesPerFrame(k);
    else
        totalTimes(k) = timesPerFrame(k) + totalTimes(k-1);
    end
end

bestForeheadPixelRegion = getBestPixelRegions(foreheadPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 10, 1, 80);
bestLeftCheekPixelBox = getBestPixelRegions(leftCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 10, 1, 80);
bestRightCheekPixelBox = getBestPixelRegions(rightCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 10, 1, 80);


i = numOfInitialFrames; % starting at one index in front of the initial frames for the new data
for numOfIterations = 1:1 
    i = i + 1;
    length_HR = length(HR); % used for debugging
    
    % Getting new PPG data
    for i = i:(i+es-1)
        tic
        img = snapshot(cam);
    if mod(i,5) == 0 || i == 1
        [roiForehead{i}, roiLeftCheek{i}, roiRightCheek{i}] = detectCheeisAndForehead_V2(img);
    else
        roiForehead{i} = roiForehead{i-1};
        roiLeftCheek{i} = roiLeftCheek{i-1};
        roiRightCheek{i} = roiRightCheek{i-1};
    end 
    if roiForehead{i}==1
        if i>1 
            roiForehead{i} = roiForehead{i-1};
            roiLeftCheek{i} = roiLeftCheek{i-1};
            roiRightCheek{i} = roiRightCheek{i-1};
        else 
            faceRoi = [250,150;300,250;250,250;300,150];
            x = faceRoi(:,1);
            y = faceRoi(:,2);
            faceWidth = max(x) - min(x);
            faceHeight = max(y) - min(y);
            
            roiForehead{i} = [250,min(y) + 0.5*faceHeight;300,min(y) + 0.75*faceHeight;250,min(y);300,min(y)];
            
            leftCheekX = [min(x); min(x); min(x) + faceWidth*0,4; min(x) + faceWidth*0.4];
            leftCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
            roiLeftCheek = [leftCheekX leftCheekY];

            rightCheekX = [min(x) + faceWidth*0.6; min(x) + faceWidth*0.6; max(x); max(x)];
            rightCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
            roiRightCheek = [rightCheekX rightCheekY];
        end
        foreheadPixelBoxArrays(i,1).pixelBoxArray = roiToPixelBoxArray(roiForehead{i}, img);
        leftCheekPixelBoxArrays(i,1).pixelBoxArray = roiToPixelBoxArray(roiLeftCheek{i}, img);
        rightCheekPixelBoxArrays(i,1).pixelBoxArray = roiToPixelBoxArray(roiRightCheek{i}, img);
        
        t = tic;
        timesPerFrame(i) = toc(t);

        if i == 1
            totalTimes(i) = timesPerFrame(i);
        else
            totalTimes(i) = timesPerFrame(i) + totalTimes(i-1);
        end
    end
    end
    
    % getting each pixelBox's PSDs, goodness metrics, and returning the top
    % pixel boxes according to a preset goodness metric threshold
    
    bestForeheadPixelRegion = getBestPixelRegions(foreheadPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 1, 1, 80);
    bestLeftCheekPixelBox = getBestPixelRegions(leftCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 1, 1, 80);
    bestRightCheekPixelBox = getBestPixelRegions(rightCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 1, 1, 80);
    
    
%     This is all in getBestPixelRegions
%     % detrend
%     r_detrend = detrend(r);
%     b_detrend = detrend(b);
%     g_detrend = detrend(g);
%     
%     % normalize
%     r_norm = normalize(r_detrend);
%     b_norm = normalize(b_detrend);
%     g_norm = normalize(g_detrend);

    % The ICA stuff is in getPixelRegions function
    % ICA feature selection OR PCA
    % X = [totalTimes; r_norm; b_norm; g_norm];
    % X = [totalTimes; timesPerFrame; r_norm; b_norm; g_norm]; 
    % X = [totalTimes; timesPerFrame; r_detrend; b_detrend; g_detrend]; 
    % X = [totalTimes; timesPerFrame; r_norm; b_norm; g_norm]; 
    % [pulse_ica, W, T, mu] = kICA(X,5); % changed to 3 source, find best PPG signal

    
%     In getBestPixelRegions function
%     figure(1)
%     plot(freq,10*log10(psdx))
%     grid on
%     title('Periodogram Using FFT')
%     xlabel('Frequency (Hz)')
%     ylabel('Power/Frequency (dB/Hz)')

    % Best component selection
    % best_comp = 5; % green channel
    % Xb = X(5,:); % Xb = X([1 5],:); % took out totalTimes b/c no current face detection
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
    
    %{
    Notes for Future:
    1. after two weeks, interpolate data after normalizing data and before PCA
    2. People used ICA, and these sources were used after applying data
        - research fastICA algorithm
        - If given three sources, it will make 3 features, and with PSD,
        find the highese peak with each feature, and whichever has the
        highest peak is used for the rest of the algorithm
    %}
    
    
    % graph pulse_sw variable against time to make sure that it makes sense
    
    % graph Xb against time to see overlay of both - include this in the
    % paper to demonstrate moving average

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
    
    % HR
    %%%%%%%%%%%%%%%%%
    
    
    % Loop Frames % if the number of initial frames + es is the limit to
    % the frames that one processes at once, then this if statement will be
    % no longer needed
    if i == numOfInitialFrames + es
        i = numOfInitialFrames;
        % resizing data to have newest 200 frames of data by taking out the
        % first 20 frames, which reduces these arrays of 220 elements to 200
        % elements

        % r = reduceToLastNIndices(r, numOfInitialFrames);
        % g = reduceToLastNIndices(g, numOfInitialFrames);
        % b = reduceToLastNIndices(b, numOfInitialFrames);
        % timesPerFrame = reduceToLastNIndices(timesPerFrame, numOfInitialFrames);
        % totalTimes = reduceToLastNIndices(totalTimes, numOfInitialFrames);
        
    end
end
    
% Save Data
% save('data_Initials_video#.mat','r','g','b');