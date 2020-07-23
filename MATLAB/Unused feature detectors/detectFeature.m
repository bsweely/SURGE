% Detects a facial feature as specified in the argument
% the feature argument is a string of the name of a facial feature that is
% already detectable according to the vision.CascadeObjectDetector()
% method. The img argument is the image file made from the raspberry pi.

function roi = detectFeature(img, feature)
detector = vision.CascadeObjectDetector(feature);
bboxes = step(detector,img);
Ifeatures=insertObjectAnnotation(img, 'rectangle', bboxes, feature);
if ~isempty(bboxes)
    roi = bbox2points(bboxes);
else
    roi = 1;
end
imagesc(Ifeatures), title(strcat('Detected_', feature)), drawnow;
end
