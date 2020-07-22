clear; close all; % clc;

% This is the MRC file with the face tracking stuff in it. V4 is a backup
% of the latest MRC file without face tracking to make sure that if the
% face tracking code is a mistake, we can easily go back to a non-tracking
% version of MRC.

%{
mypi=raspi('IP Address','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
%}

load('Jeremy_data_1_HR67_.mat');

end_sample=20; % set how many seconds you want to loop
es = length(images);
Fs = floor(fps);
roiC = cell(Fs);
iterationCount = floor(length(images)/es);
iterationRange = 0:iterationCount-1;
r = zeros(1,es);
g = zeros(1,es);
b = zeros(1,es);

timesPerFrame = zeros(1,es);
totalTimes = zeros(1,es);
HR = [];

% The MRC MATLAB code that I am reading does faceial recognition before
% transforming the images for face tracking, but in this code, the images
% are transformed so that the rois have the correct coordinates when it
% happens. Perhaps we should do it in their order to test performance. 

%% Processing image data

% getting image data

img = cell(1,es);
imgGray = cell(1,es);
points = cell(1,es);
pointsFount = cell(1,es);
tracker = vision.PointTracker('MaxBidirectionalError', 2);
for iteration = iterationRange
    
    % initializing the face tracker
    
    for i = (iteration*es)+1:(iteration+1)*es
        % i
        tic
        % img{i} = snapshot(cam);

        img{i} = images(i).snapshot;
        imgGray{i} = rgb2gray(images(i).snapshot);
        rois = detectbothcheeks_V4(img{i});
        roiC{i, 1} = rois{1}; % gets wrong each time that i == 82 with "ambiguous dimension" error
        roiHead = rois{2};
        bbox = points2bbox(roiHead); % bbox of face before transformation
        
        % Face tracking transformation
        points{i} = detectMinEigenFeatures(imgGray{i}, 'roi', bbox); % Getting the points for tracking
        points{i} = points{i}.Location;
        initialize(tracker, points{i}, imgGray{i});
        
        % Repositioning faces based on the first frame
        if i ~= i
            % Reposition faces
        [points{i-1}, pointsFound] = step(tracker, ImgGray{i});
        pointsInFrame = points{i}(pointsFound, :); % This line of code might be misunderstood by me
        oldPointsInFrame = points{i-1}(pointsFound, :);
        
        numberOfPointsFound = length(pointsInFrame);
        
        if numberOfPointsFound >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [imageTransform, oldPointsInFrame, pointsInFrame] = estimateGeometricTransform(...
                oldPointsInFrame, pointsInFrame, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box points
            newBboxPoints = transformPointsForward(imageTransform, bbox);

            % Reset the points
            points{i} = pointsInFrame;
            setPoints(tracker, points{i});
        end
        end
    end
end

% Transforming images to overlap each other well for analysis
tracker = vision.PointTracker('MaxBidirectionalError', 2);


% Deriving the ROIs from the whole face
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

%% Processing signal data

bandpass = [0.8 3];
pulseRateError = 5;
numBestBoxes = 10;
estimatedPulseRate = 70;
bestForeheadPixelRegion = getBestPixelRegions_V2(foreheadPixelBoxArrays, Fs, bandpass,...
    pulseRateError, numBestBoxes, estimatedPulseRate);

    
% Save Data
% save('data_Initials_video#.mat','r','g','b');