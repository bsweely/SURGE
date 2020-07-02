% Loops through to show the types of ClassificationModels in computer
% facial recognition in MATLAB
clear; close all; clc;

% list of all models
models = ["FrontalFaceCART" "FrontalFaceLBP" "UpperBody" "EyePairBig" "EyePairSmall" "LeftEye" "RightEye" "LeftEyeCART" "RightEyeCART" "ProfileFace" "Mouth" "Nose"];

% Initializing the Raspberry Pi
Fs = 15; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20; % If I set this to be es = 10, then it has a syntax error?
roi = cell(Fs,1);

% Variables that are not needed in this code as for now
% rTotal = zeros(1,50);
% bTotal = zeros(1,50);
% gTotal = zeros(1,50);
% HR = [];

numberOfImages = 5;
images = cell(1, numberOfImages);

for k = 1:numberOfImages
    images{k} = snapshot(cam);
end

% The following two bested loops iterate as follows:
% For each image in the collection of face images we are analyzing for
% facial features, do the following:
% for each facial feature as named in the facial features models list
% above, detect the facial feature, cocnvert it into a PDF, and name it
% after the image and facial featuer in it. 
for j = 1:numberOfImages
    % img = snapshot(cam);
    pdfsList = cell(1,numberOfImages);
    for i = 1:length(models)
    %{
    if i == 1
        roi{i} = detectfaces_V2(img)
        i = i + 1;
    else
        roi{i} = detectFeature(img, models(i))
    end
    %}
        figure(i);
        roi{i} = detectFeature(images{j}, models(i));
        figName = strcat(int2str(j), "Detected", models(i), ".pdf");
        saveas(gcf, figName);
%         export_fig figName;
%         pdfsList{i} = figName
    end
%     nameOfCombinedFile = strcat("Image", int2str(j), ".pdf")
%     append_pdfs nameOfCombinedFile pdfsList;
%     save(nameOfCombinedFile)
end
    
    
    
    