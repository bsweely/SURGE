% Detecting the eyes and nose to make the ROI
function [roiForehead, roiLeftCheek, roiRightCheek] = detectCheeksAndForehead_V2(img)
% detectfaces_V2.m code
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector,img);
Ifaces=insertObjectAnnotation(img, 'rectangle', bboxes, 'Face');
if ~isempty(bboxes)
    roiHead = bbox2points(bboxes);
    
    % If there are multiple faces present in the frame, the collowing line
    % takes the biggest (and therefore the real) face and sets the rois from
    % it.
    roiHead = getBiggestROI(roiHead);
    imagesc(Ifaces), title('Detected faces'), drawnow;
    
    % isolating the forehead

    % Forehead-only Data
    % Online, multiple art websites and facial recognition sites said the
    % forehead starts about halfway up the face and ends 3/4 up the face. 
    % The forehead spans about the same width of the face, and even more so
    % for some people, so the x coordinates for the forehead ROI are the
    % same as the face, but the y coordinates are limited to the upper 3/4
    % of the face only.

    x = roiHead(:,2);
    y = roiHead(:,1);

    faceHeight = max(y) - min(y);
    bottomOfFH = round(min(y) + faceHeight*0.5); % approximately the height of bottom of forehead
    topOfFH = round(min(y) + faceHeight*0.75); % approximately the height of top of forehead

    roiForeheadX = [min(x); min(x); max(x); max(x)];
    roiForeheadY = [bottomOfFH; topOfFH; bottomOfFH; topOfFH];
    roiForehead = [roiForeheadX roiForeheadY];
    drawROI(roiForehead, img, 'roiForehead')

    % isolating cheeks

    faceWidth = max(x) - min(x);
    faceHeight = max(y) - min(y);

    leftCheekX = [min(x); min(x); min(x) + faceWidth*0.4; min(x) + faceWidth*0.4];
    leftCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
    roiLeftCheek = [leftCheekX leftCheekY];
    drawROI(roiLeftCheek, img, 'roiLeftCheek')

    rightCheekX = [min(x) + faceWidth*0.6; min(x) + faceWidth*0.6; max(x); max(x)];
    rightCheekY = [min(y); min(y) + faceHeight*0.5; min(y); min(y) + faceHeight*0.5];
    roiRightCheek = [rightCheekX rightCheekY];
    drawROI(roiRightCheek, img, 'roiRightCheek')
else
    roiForehead = 1;
    roiLeftCheek = 1;
    roiRightCheek = 1;
end 
end