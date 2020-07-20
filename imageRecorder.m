clear; close all; clc;

Fs = 30; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
roi = cell(Fs,1);
roiCheeks = cell(Fs,1);
numFrames = 500;
name = input("Enter name with tick marks like 'john' or 'emily'. When your name is typed, look into the camera and press enter, and the program will start. Enter name: ");
numTime = input("Type the file number: ")
fileName = strcat(name, '_data_', numTime, '_');
images = struct();

% getting the videos for data collection
t = tic;
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
    pause(1/Fs)
end
time = toc(t); % time elapsed
fps = numFrames/time; % calculated frame rate

disp('500 done. Please wait as the file processes...')
HR = input("Please enter the pulse oximeter heart rate if you have one as 'HR'. If you don't have one, just type '' instead: ")
% saving the .mat file
fileName = strcat(fileName, 'HR', HR, '_.mat');
save(fileName, 'images','fps')

disp(strcat('Check the "current folder" to the right to see if a .mat file named: ',fileName,...
            ' Is present. If so, please put this .mat file into the data folder into material -> Matlab Code -> new data folder. The program is finished.'))



