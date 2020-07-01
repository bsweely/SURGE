% MRC Algorithm Drafting

% This file has the algorithms for calculating PPG through algorithms that
% are described and coded in the technical literature


% Algorithm 1: distancePPG Region Tracker Algorithm
% Paper: Kumar
% Purpose: Tracking facial features and resetting the deformable facial
% recognition model when the time interval for replacement is achieved
% (say, every 20 seconds, or ecery 30 seconds). We should write the
% algorithm to reset based on a number of frames that pass or indicated
% movement of the baby.
% parameters: V(x,y,t) is the intensity of the images, according to the
% paper. We should make this be the collection of images or a single image.
% Returns: The tracked PPG regions for HR detection and processing

function PPGRegions = TrackPPGRegion(V(x,y,t))
t = 0;
T = 20; % 20 second until the deformable face model and KLT algorithm reset
while t <= T
    if mod(t,T).*FPS == 0
        % Restart the loop. This condition restarts it every T seconds
        
        % Initialize the landmark points of the face for tracking:
        LP = DeformableFace(V(x,y,t)); % DeformableFace is the function that detects features and returns points
        
        % Isolate the 7 planar regions on the face (3 forehead, both
        % cheeks, 2 on chin):
        PlanarRegions(t) = DefinePlanarRegions(LP);
        
        % Define the PPG ROIs for the t<th> frame
        PPGRegions(t) = DefinePPGRegion(PlanarRegions(t));
        
        % Clean the PPGRegions data by filtering out the
        % erroneous points and keeping the stable ones: 
        GoodPPGRegions(0) = GoodFeatures(V(x,y,t), PlanarRegions(t));
        
    else
        % Resetting the KLT Tracker 
        GoodPPGRegions(1) = TrackerKLT(V(x,y,t), V(x,y,t-1), GoodPPGRegions(0));
        
        % AM(t) refers to the affine model parameter computed separately
        % for each planar region
        AM(t) = AffineRANSAC(GoodPPGRegions(1), GoodPPGRegions(0));
        
        % Not sure what this does:
        PlanarRegions(t) = PlanarRegions(t-1)*AM(t);
        % Each PPG region in PPGRegions tracked with an encompassing planar
        % region motion model
        PPGRegions(t) = PPGRegions(t-1) % *AM(t)^p power? What is this?
        
        GoodPPGRegions(0) = GoodPPGRegions(1);
        
    end
    t = t + 1;
end 

return PPGRegions % returns tracked PPG regions' locations

end
        
        



