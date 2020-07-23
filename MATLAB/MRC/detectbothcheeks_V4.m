function rois = detectbothcheeks_V3(img)
% detectfaces_V2.m code
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector,img);
if ~isempty(bboxes)
    roiHead = bbox2points(bboxes); 
    roiHead = getBiggestROI(roiHead);
    
    x = roiHead(:,1);
    y = roiHead(:,2);
    roii = cell(3,1);
    faceWidth = max(x) - min(x);
    faceHeight = max(y) - min(y);
    A = sqrt(faceWidth*faceHeight*0.04); % area that scales with face
    
    bottomOfFH = round(min(y)); % approximately the height of bottom of forehead
    topOfFH = round(min(y) + faceHeight*0.2); % approximately the height of top of forehead

    roiForeheadX = ((min(x) + max(x)) / 2);
    roiForeheadY = ((bottomOfFH + topOfFH) / 2);
    roiF = [roiForeheadX roiForeheadY]; % forehead center coordinates
    roii{1} = [roiF(1)-A roiF(2)+A/2; roiF(1)-A roiF(2)-A/2; roiF(1)+A roiF(2)+A/2; roiF(1)+A roiF(2)-A/2];

    % isolating cheeks
    leftCheekX = (min(x) + faceWidth*0.3);
    leftCheekY = (min(y) + faceHeight*0.65);
    roiL = [leftCheekX leftCheekY];
    roii{2} = [roiL(1)-A/2 roiL(2)+A/2; roiL(1)-A/2 roiL(2)-A/2; roiL(1)+A/2 roiL(2)+A/2; roiL(1)+A/2 roiL(2)-A/2];

    rightCheekX = (min(x) + faceWidth*0.7);
    rightCheekY = (min(y) + faceHeight*0.65);
    roiR = [rightCheekX rightCheekY];
    roii{3} = [roiR(1)-A/2 roiR(2)+A/2; roiR(1)-A/2 roiR(2)-A/2; roiR(1)+A/2 roiR(2)+A/2; roiR(1)+A/2 roiR(2)-A/2];
    
    bboxes = [roii{1}(1,1) roii{1}(2,2) 2*A A; roii{2}(1,1) roii{2}(2,2) A A; roii{3}(1,1) roii{3}(2,2) A A];
    Ifaces=insertObjectAnnotation(img, 'rectangle', bboxes, 'ROI');
    imagesc(Ifaces), title('Detected forehead'), drawnow;
else
    roii{1} = 1;
    roii{2} = 1;
    roii{3} = 1;
    roiHead = 1;
end
rois = cell(1,2);
rois = {roii, roiHead};
end