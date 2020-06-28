% Detecting the right eye of the face

function roi = detectrighteye(img)
rightEyeDetector = vision.CascadeObjectDetector('RightEye');
bboxes = step(rightEyeDetector,img);
IrightEyes=insertObjectAnnotation(img, 'rectangle', bboxes, 'Right Eye');
if ~isempty(bboxes)
    roi = bbox2points(bboxes);
else
    roi = 1;
end
imagesc(IrightEyes), title('Detected Right Eye'), drawnow;
end





