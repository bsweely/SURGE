clear; close all; % clc;

Fs = 90; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

%{
mypi=raspi('IP Address','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
%}

load('Jeremy_data_10_98_.mat');

end_sample=20; % set how many seconds you want to loop
es = 50;
roiC = cell(Fs);
numOfInitialFrames = length(images) - 150; %length(images); % number of initial frames to acquire before moving average % set to 10 to debug faster
iterationCount = floor(length(images)/es);
iterationRange = 0:iterationCount-1;
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
img = cell(1,numOfInitialFrames);
for iteration = iterationRange
    for i = (iteration*es)+1:(iteration+1)*es
        % i
        tic
        % img{i} = snapshot(cam);

        img{i} = images(i).snapshot;
        roiC{i} = detectbothcheeks_V4(img{i});

        % collecting timestamps for images
        t = tic;
        timesPerFrame(i) = toc(t);
        if i == 1
            totalTimes(i) = timesPerFrame(i);
        else
            totalTimes(i) = timesPerFrame(i) + totalTimes(i-1);
        end
    end
end

% processing data

clear i iteration;

for iteration = iterationRange
    for i = (iteration*es)+1:(iteration+1)*es
        roif = roiC{i}{1};
        roil = roiC{i}{2};
        roir = roiC{i}{3};
        if roif==1
            if i>1 
                roif = roiC{i-1}{1};
                roil = roiC{i-1}{2};
                roir = roiC{i-1}{3};
            else 
                roif = [250,150;300,250;250,250;300,150];
                roil = [250,150;300,250;250,250;300,150];
                roir = [250,150;300,250;250,250;300,150];
                roiC{i}{1} = [250,150;300,250;250,250;300,150];
                roiC{i}{2} = [250,150;300,250;250,250;300,150];
                roiC{i}{3} = [250,150;300,250;250,250;300,150];
            end
        end
        
        
        roif = round(roif);
        roil = round(roil);
        roir = round(roir);
        
        %{
        xf = roif(:,1)
        yf = roif(:,2)
        xl = roil(:,1)
        yl = roil(:,2)
        xr = roir(:,1)
        yr = roir(:,2)
        
        
        froi = img{i}(min(xf):max(xf),min(yf):max(yf),:)
        lroi = img{i}(min(xl):max(xl),min(yl):max(yl),:)
        rroi = img{i}(min(xr):max(xr),min(yr):max(yr),:)
        %}
        
        
        froi = img{i}(round(roif(2,2)):round(roif(1,2)), round(roif(1,1)):round(roif(3,1)),:);
        lroi = img{i}(round(roil(2,2)):round(roil(1,2)), round(roil(1,1)):round(roil(3,1)),:);
        rroi = img{i}(round(roir(2,2)):round(roir(1,2)), round(roir(1,1)):round(roir(3,1)),:);
        
        
        f = [roif(2,2) roif(1,1)
        roif(2,2) roif(3,1)
        roif(1,2) roif(1,1)
        roif(1,2) roif(3,1)];

        l = [roil(2,2) roil(1,1)
        roil(2,2) roil(3,1)
        roil(1,2) roil(1,1)
        roil(1,2) roil(3,1)];

        r = [roir(2,2) roir(1,1)
        roir(2,2) roir(3,1)
        roir(1,2) roir(1,1)
        roir(1,2) roir(3,1)];

        figureNumbers = [1 2 3 4 5 6 7];
        figureTitle = {'Forehead 20x20', 'Left cheek 20x20', 'Right cheek 20x20'};

        foreheadPixelBoxArrays(i).frame = roiToArrayOfPixelBoxes(f, img{i}, figureNumbers(1), figureTitle(1));
        leftCheekPixelBoxArrays(i).frame = roiToArrayOfPixelBoxes(l, img{i}, figureNumbers(3), figureTitle(3));
        rightCheekPixelBoxArrays(i).frame = roiToArrayOfPixelBoxes(r, img{i}, figureNumbers(2), figureTitle(2));
    end
end



bandpass = [0.8 3];
pulseRateError = 5;
numBestBoxes = 10;
estimatedPulseRate = 80;
bestForeheadPixelRegion = getBestPixelRegions(foreheadPixelBoxArrays, Fs, totalTimes, timesPerFrame,...
    bandpass, pulseRateError, numBestBoxes, estimatedPulseRate);

    
% Save Data
% save('data_Initials_video#.mat','r','g','b');