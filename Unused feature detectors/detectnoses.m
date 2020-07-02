% detects the nose and returns the roi of the nose
function roi = detectnoses(img)
noseDetector = vision.CascadeObjectDetector('Nose');
bboxes = step(noseDetector,img);
Inoses=insertObjectAnnotation(img, 'rectangle', bboxes, 'Nose');
if ~isempty(bboxes)
    roi = bbox2points(bboxes);
else
    roi = 1;
end
imagesc(Inoses), title('Detected noses'), drawnow;
end
