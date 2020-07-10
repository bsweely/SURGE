clear; close all; clc;

Fs = 30; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);
roiCheeks = cell(Fs,1);
numFrames = 500;
timesPerFrame = zeros(1,numFrames);
totalTimes = zeros(1,numFrames);
name = input('Enter name: ');
x = zeros(4,numFrames);
y = zeros(4,numFrames);
r = zeros(1,numFrames);
g = zeros(1,numFrames);
b = zeros(1,numFrames);
roiValues = struct();

% getting the videos for data collection
for i = 1:numFrames
    img = snapshot(cam);
    tic;
    roi{i} = detectfaces_V2(img);
    if roi{i}==1
        roi{i} = [250,150;300,250;250,250;300,150];
    end
    
    x = roi{i}(:,2);
    y = roi{i}(:,1);
    Red_ROI = img(min(x(i)):max(x(i)),min(y):max(y),1);
    Blue_ROI = img(min(x(i)):max(x(i)),min(y):max(y),3);
    Green_ROI = img(min(x(i)):max(x(i)),min(y):max(y),2);
    r(i) = sum(sum(Red_ROI)); % intensity -> PPG
    b(i) = sum(sum(Blue_ROI));
    g(i) = sum(sum(Green_ROI));
    
    t = tic
    timeElapsed = toc(t);
    timesPerFrame(i) = timeElapsed;
    if i == 1
        totalTimes(i) = timesPerFrame(i);
    else
        totalTimes(i) = totalTimes(i-1) + timesPerFrame(i);
    end

    % Forehead-only Data
    % Online, multiple art websites and facial recognition sites said the
    % forehead starts about halfway up the face and ends 3/4 up the face. 
    % The forehead spans about the same width of the face, and even more so
    % for some people, so the x(i) coordinates for the forehead ROI are the
    % same as the face, but the y coordinates are limited to the upper 3/4
    % of the face only.
    faceHeight = max(y(i)) - min(y(i));
    bottomOfFH = round(min(y(i)) + faceHeight*0.5); % approx(i)imately the height of bottom of forehead
    topOfFH = round(min(y(i)) + faceHeight*0.75); % approx(i)imately the height of top of forehead

    Red_ROI_FH = img(min(x(i)):max((i)),bottomOfFH:topOfFH,1);
    Blue_ROI_FH = img(min(x(i)):max(x(i)),bottomOfFH:topOfFH,3);
    Green_ROI_FH = img(min(x(i)):max(x(i)),bottomOfFH:topOfFH,2);
    rFH(i) = sum(sum(Red_ROI_FH));
    bFH(i) = sum(sum(Blue_ROI_FH));
    gFH(i) = sum(sum(Green_ROI_FH));
end

% Saving the data as mat files
% Full-face data
%{
fileEx(i)tString = '.mat';
filenameString = strcat(name, '_data', fileEx(i)tString);
save(filenameString, 'r', 'g', 'b', 'rFH', 'gFH', 'bFH', );
%}






