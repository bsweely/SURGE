function roiData = roiToRGB(roi, img)
% getting x, y, and RGB values from an roi from an image
roiData = struct();
roiData.x = roi(:,1);
roiData.y = roi(:,2);
Red_ROI = img(min(roiData.x):max(roiData.x),min(roiData.y):max(roiData.y),1);
Blue_ROI = img(min(roiData.x):max(roiData.x),min(roiData.y):max(roiData.y),3);
Green_ROI = img(min(roiData.x):max(roiData.x),min(roiData.y):max(roiData.y),2);
roiData.r = sum(sum(Red_ROI)); % intensity -> PPG
roiData.b = sum(sum(Blue_ROI));
roiData.g = sum(sum(Green_ROI));
end