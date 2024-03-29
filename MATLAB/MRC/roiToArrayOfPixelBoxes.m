function frame = roiToArrayOfPixelBoxes(roi, img, figureNumber, figureTitle)
% This function returns the coordinates for pixel boxes, the goodness
% metrics, and other parts of choosing which regions to analyze with PPG

% Defining the struct for each pixel box

%{
Each pixel box has the roi coordinates, goodness metric, and R, G, and B
intensities. It also has the power spectral densities and the peaks that
are associated with them.

Current Errors:
This function includes regions that definitely are not helpful, like the
mouth and eyeball. This is a future direction for this code. 
%}

pixelBoxInstance = struct();
pixelBoxInstance.roiCoords = [];
pixelBoxInstance.bbox = [];
pixelBoxInstance.rIntensity = 0;
pixelBoxInstance.gIntensity = 0;
pixelBoxInstance.bIntensity = 0;


x = floor(roi(:,2));
y = floor(roi(:,1));


yPixelBoxBounds = min(y):20:max(y); % The y coordinates of the 20x20 pixels
xPixelBoxBounds = min(x):20:max(x); % The x coordinates of the 20x20 pixels
numOfPixelBoxes = (length(yPixelBoxBounds) - 1).*(length(xPixelBoxBounds) - 1); % number of pixel boxes
frame(numOfPixelBoxes) = struct();
bboxes = [];

index = 0;
% initializing the pixelBox objects
for col = 1:length(yPixelBoxBounds)
    for row = 1:length(xPixelBoxBounds)
        index = index + 1;
        
        Red_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,1); 
        Green_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,2); 
        Blue_ROI = img(xPixelBoxBounds(row):xPixelBoxBounds(row)+20,yPixelBoxBounds(col):yPixelBoxBounds(col)+20,3); 
        
        pixelBoxInstance.rIntensity = sum(sum(Red_ROI)); % intensity -> PPG
        pixelBoxInstance.gIntensity = sum(sum(Green_ROI));
        pixelBoxInstance.bIntensity = sum(sum(Blue_ROI));
        
        % initializing roi coords
        pixelBoxInstance.roiCoords = [xPixelBoxBounds(row) yPixelBoxBounds(col)
            xPixelBoxBounds(row) (yPixelBoxBounds(col)+20)
            (xPixelBoxBounds(row)+20) yPixelBoxBounds(col)
            (xPixelBoxBounds(row)+20) (yPixelBoxBounds(col)+20)];
        
        pixelBoxInstance.bbox = points2bbox(pixelBoxInstance.roiCoords);
        bboxes = [bboxes; pixelBoxInstance.bbox]; 
        
        % concatenating the current pixelboxInstance to the pixel box array
        frame(index).pixelBoxInstance = pixelBoxInstance;
    end
end
figure(figureNumber)
Ifaces=insertObjectAnnotation(img, 'rectangle', bboxes, '');
imagesc(Ifaces), title(figureTitle), drawnow;
end





