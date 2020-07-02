% Detecting the eyes and nose to make the ROI
function roi = detectbothcheeks(img)
quitSignal = 0;
while quitSignal == 0
    noseDetector = vision.CascadeObjectDetector('Nose');
    eyesDetector = vision.CascadeObjectDetector('EyePairBig');
    bboxesNose = step(noseDetector,img);
    bboxesEyes = step(eyesDetector,img);
    Inoses=insertObjectAnnotation(img, 'rectangle', bboxesNose, 'Nose');
    Ieyes=insertObjectAnnotation(img, 'rectangle', bboxesEyes, 'Eyes');
    if ~isempty(bboxesNose)
        roiNose = bbox2points(bboxesNose);
    else
        roiNose = 1;
    end
    if ~isempty(bboxesEyes)
        roiEyes = bbox2points(bboxesEyes);
    else
        roiEyes = 1;
    end
    if ~isequal(roiNose,1) && ~isequal(roiEyes,1)
        quitSignal = 1;
        imagesc(Inoses), title('Detected noses'), drawnow;
        imagesc(Ieyes), title('Detected eyes'), drawnow;
    else
        continue
    end
    
    % Mathematically making the cheek ROI
    
    % Getting pixel values for both rois
    xNose = roiNose(:,1);
    yNose = roiNose(:,2);
    
    xEyes = roiEyes(:,1);
    xEyes = roiEyes(:,2);
    
    % Getting the coordinates for the right cheek
    % The following coordinates are best illustrated with a face drawing
    % that I will include in the material file and on Github if it is
    % possible so that you see the coordinates that I have here. 
    
    A = [min(xEyes),max(yNose)];
    B = [min(xNose),max(yNose)];
    C = [min(xEyes),min(yNose)];
    D = [min(xNose),min(yNose)];
    
    % These are the coordinates of the left cheek ROI following the same
    % idea as the right cheek ROI
    
    E = [max(xNose),max(yNose)];
    F = [max(xEyes),max(yNose)];
    G = [max(xEyes),min(yNose)];
    H = [max(xNose),min(yNose)];
    
    % making the boxes for the cheek ROIs 
    roiRightCheek = [C; B; A; D];
    roiLeftCheek = [H; F; E; G];
    
    % combining the rois
    % horzcat requires the number of rows to be the same for the matrices
    % that are being combined, and because the heights of both rois are
    % determined by the same y coordinates (min(yNose) and max(yNose)),
    % this will not be a problem here
    roi = horzcat(roiRightCheek, roiLeftCheek);
    
    
end
end
    
