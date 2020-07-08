function biggestDetectedROI = getBiggestROI(roi)

areas = [];
roiSize = size(roi);
roiDimensions = length(size(roi));
numOfFaces = 0;
if roiDimensions == 2
    biggestDetectedROI = roi
else % if roiDimensions >= 3
    numOfFaces = roiSize(3);
    for i = 1:numOfFaces
        areas(i) = (max(roi(:,2,i)) - min(roi(:,2,i))).*(max(roi(:,1,i)) - min(roi(:,1,i)));
    end
    
    indexOfBiggestArea = find(areas == max(areas))
    roi = roi(:,:,indexOfBiggestArea);
end

biggestDetectedROI = roi;

end

