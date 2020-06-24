function roi = detectfaces_V2(img)
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector,img);
Ifaces=insertObjectAnnotation(img, 'rectangle', bboxes, 'Face');
if ~isempty(bboxes)
    roi = bbox2points(bboxes);
else
    roi = 1;
end
imagesc(Ifaces), title('Detected faces'), drawnow;
end