% Detecting the left eye of the face

function roi = detectlefteye(img)
leftEyeDetector = vision.CascadeObjectDetector('LeftEye');
bboxes = step(leftEyeDetector,img);
IleftEyes=insertObjectAnnotation(img, 'rectangle', bboxes, 'Right Eye');
if ~isempty(bboxes)
    roi = bbox2points(bboxes);
else
    roi = 1;
end
imagesc(IleftEyes), title('Detected Left Eye'), drawnow;
end