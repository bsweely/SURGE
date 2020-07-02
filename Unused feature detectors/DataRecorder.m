qclear; close all; clc;

Fs = 15; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);

for j = 1:10 % getting the videos for data collection
    for i = 1:es
    img = snapshot(cam);
    roi{i} = detectfaces_V2(img);
    if roi{i}==1
        if i>1 
            roi{i} = roi{i-1};
        else 
            roi{i} = [250,150;300,250;250,250;300,150];
        end
    end
    x = roi{i}(:,2);
    y = roi{i}(:,1);
    Red_ROI = img(min(x):max(x),min(y):max(y),1);
    Blue_ROI = img(min(x):max(x),min(y):max(y),2);
    Green_ROI = img(min(x):max(x),min(y):max(y),3);
    r(i) = sum(sum(Red_ROI)); % intensity -> PPG
    b(i) = sum(sum(Blue_ROI));
    g(i) = sum(sum(Green_ROI));
    
    % Forehead-only Data
    % Online, multiple art websites and facial recognition sites said the
    % forehead starts about halfway up the face and ends 3/4 up the face. 
    % The forehead spans about the same width of the face, and even more so
    % for some people, so the x coordinates for the forehead ROI are the
    % same as the face, but the y coordinates are limited to the upper 3/4
    % of the face only.
    faceHeight = max(y) - min(y);
    bottomOfFH = round(min(y) + faceHeight*0.5); % approximately the height of bottom of forehead
    topOfFH = round(min(y) + faceHeight*0.75); % approximately the height of top of forehead
    
    Red_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,1);
    Blue_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,3);
    Green_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,2);
    rFH(i) = sum(sum(Red_ROI_FH));
    bFH(i) = sum(sum(Blue_ROI_FH));
    gFH(i) = sum(sum(Green_ROI_FH));
    
    % Saving the data as mat files
    % Full-face data
    nameString = 'data_JS_';
    numString = int2str(j);
    fileExtString = '.mat';
    filenameString = strcat(nameString, numString, fileExtString);
    save(filenameString, 'r', 'g', 'b');
    
    % Forehead-only data
    filenameStringFH = strcat('FH_', filenameString);
    save(filenameStringFH, 'rFH', 'bFH', 'gFH');
    
    
    
    
    
    end
end