clear; close all; clc;

Fs = 30; % framerate needs to be higher 
% home
mypi=raspi('IP ADDRESS','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);
roiCheeks = cell(Fs,1);
numFrames = 500;
timesPerFrame = zeros(1,numFrames);
totalTimes = zeros(1,numFrames);
name = input("Enter name with tick marks like 'john' or 'emily'. When your name is typed, look into the camera and press enter, and the program will start. Enter name: ");
numTime = input("Type '1' if this is the first of two files for you, or type '2' if this is the second file: ")
fileName = strcat(name, '_data_', numTime, '_');
images = struct();

% getting the videos for data collection
for i = 1:numFrames
    images(i).snapshot = snapshot(cam);
    
    if i == 1
        disp('Program has started. Wait until you see "500 done..." in the command line...')
    elseif i == 100
        disp('100 done...')
    elseif i == 200
        disp('200 done...')
    elseif i == 300
        disp('300 done...')
    elseif i == 400
        disp('400 done...')
    end
end

disp('500 done. Please wait as the file processes...')
HR = input("Please enter the pulse oximeter heart rate if you have one as 'HR'. If you don't have one, just type '' instead: ")
% saving the .mat file
fileName = strcat(fileName, HR, '_.mat');
save(fileName, 'images')

disp(strcat('Check the "current folder" to the right to see if a .mat file named: ',fileName,...
            ' Is present. If so, please put this .mat file into the data folder into material -> Matlab Code -> new data folder. The program is finished.'))



