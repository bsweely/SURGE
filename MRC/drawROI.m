function drawROI(roi, img, title)
% This functions takes in an roi and draws it on an image and formats it
% into a MATLAB figure

bboxes = points2bbox(roi);

Iface = insertObjectAnnotation(img, 'rectangle', bboxes, 'ROI');
imagesc(Iface);
title(title); % for some reason, any string title here is 'out of bounds'. At this point, it is not worth solving this bug.
drawnow;
end