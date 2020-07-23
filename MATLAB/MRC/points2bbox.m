function bbox = points2bbox(roi)
% This is a function that takes in an roi (a set of four points) and
% outpouts a bbox (which is in for form of x,y,w,h, where x and y
% correspond to the lower left of the coordinates for the points. 

% MAKE SURE THAT YOUR POINTS MAKE A RECTANGLE! 
% THIS IS WHAT BBOXES DESCRIBE!

% Getting x and y data
x = roi(:,1);
y = roi(:,2);

% Getting width and height
width = max(x) - min(x);
height = max(y) - min(y);

% Constructing bbox
bbox = [min(x) min(y) width height];
end