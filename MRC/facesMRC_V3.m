clear; close all; % clc;

Fs = 90; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

%{
mypi=raspi('IP Address','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
%}

load('Jeremy_data_1_98_.mat');

end_sample=20; % set how many seconds you want to loop
es = 100;
roiForehead = cell(Fs,1);
roiLeftCheek = cell(Fs,1);
roiRightCheek = cell(Fs,1);

numOfInitialFrames = 100; % number of initial frames to acquire before moving average % set to 10 to debug faster
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
    % img = snapshot(cam);
    
    img = images(k).snapshot;
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
    
    foreheadPixelBoxArrays(k).frame = roiToArrayOfPixelBoxes(roiForehead{k}, img);
    leftCheekPixelBoxArrays(k).frame = roiToArrayOfPixelBoxes(roiLeftCheek{k}, img);
    rightCheekPixelBoxArrays(k).frame = roiToArrayOfPixelBoxes(roiRightCheek{k}, img);
    
    t = tic;
    timesPerFrame(k) = toc(t);

    if k == 1
        totalTimes(k) = timesPerFrame(k);
    else
        totalTimes(k) = timesPerFrame(k) + totalTimes(k-1);
    end
end

bestForeheadPixelRegion = getBestPixelRegions(foreheadPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 50, 1, 100);
bestLeftCheekPixelBox = getBestPixelRegions(leftCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 50, 1, 100);
bestRightCheekPixelBox = getBestPixelRegions(rightCheekPixelBoxArrays, Fs, totalTimes, timesPerFrame, [0.8 3], 50, 1, 100);









    
% Save Data
% save('data_Initials_video#.mat','r','g','b');