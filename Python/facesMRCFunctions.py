# Modules
import sys
import time
import argparse
import numpy as np

import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,detrend,welch
from scipy.fft import fft
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

WINDOW_NAME = 'nicu'
BBOX_COLOR = (0, 255, 0)  #green
CBOX_COLOR = (0, 0, 255) #red
FBOX_COLOR = (255, 0, 0) #blue

#parameters for analysis measured by frames
sliding_window_size = 7
global it
it = 0
global length,loop 
loop = 2
length = 20
global r,b,g
r = np.zeros((10 + loop,length))
g = np.zeros((10 + loop,length))
b = np.zeros((10 + loop,length))
r_sliding_window_avg = np.zeros(length)
g_sliding_window_avg = np.zeros(length)
b_sliding_window_avg = np.zeros(length)
roi = np.array([0,0,0,0])

# Functions from MATLAB code

# Data handling functions
# bbox2points function
def reduceToLastNIndices(array, n):
    length = array.size
    if n > length:
        print('Error: The array is not as long as the number specified')
        time.sleep(10)
        return array
    elif n == length:
        print('Error: The array is exactly as long as the number specified. Returning original array')
        time.sleep(10)
        return array
    else:
        newArray = array[(length - n):length]
        return newArray

def bbox2points(bbox):
    [x, y, w, h] = bbox
    roi = np.array([[x, y],
    [x+w, y], [x, y+h], [x+w, y+h]])

    return roi

# bbox2points function
def points2bbox(roi):
    x = roi[:, 0]
    y = roi[:, 1]

    width = max(x) - min(x)
    height = max(y) - min(y)

    bbox = np.array([min(x), min(y), width, height])

    return bbox

# getBiggestROI function
'''
This function assumes that if there are multiple faces detected, that
the roi will have a third dimension with the faces in it, as it structured
in MATLAB
'''

'''
def getBiggestROI(rois):
# This is commented out until I know the structure of an roi in the code
# Then, we can make necessary ammendments to the code from there
    areas = np.array([])
    roiSize = areas.shape
    roiDimensions = len(roiSize) # 3 dimensions for multiple faces and 2 for one face
    numOfFaces = 0
    
    if roiDimensions >= 3:
        numOfFaces = roiSize[2]
        for i in np.arange(numOfFaces):
            areas[i] = 
'''
    


# Face Detection Functions
# detectbothcheeks_V4 function
def detectbothcheeks_V3(img):
    # faceDetector = vision.CascadeObjecrtDetector
    # bboxes = step(faceDetector, img)

    if bboxes != []: # a.k.a. if bboxes is not empty...
        roiHead = bbox2points(bboxes)
        # roiHead = getBiggestROI(roiHead);

'''
function rois = detectbothcheeks_V3(img)
% detectfaces_V2.m code
faceDetector = vision.CascadeObjectDetector;
bboxes = step(faceDetector,img);
if ~isempty(bboxes)
    roiHead = bbox2points(bboxes); 
    roiHead = getBiggestROI(roiHead);
    
    x = roiHead(:,1);
    y = roiHead(:,2);
    roii = cell(3,1);
    faceWidth = max(x) - min(x);
    faceHeight = max(y) - min(y);
    A = sqrt(faceWidth*faceHeight*0.04); % area that scales with face
    
    bottomOfFH = round(min(y)); % approximately the height of bottom of forehead
    topOfFH = round(min(y) + faceHeight*0.2); % approximately the height of top of forehead

    roiForeheadX = ((min(x) + max(x)) / 2);
    roiForeheadY = ((bottomOfFH + topOfFH) / 2);
    roiF = [roiForeheadX roiForeheadY]; % forehead center coordinates
    roii{1} = [roiF(1)-A roiF(2)+A/2; roiF(1)-A roiF(2)-A/2; roiF(1)+A roiF(2)+A/2; roiF(1)+A roiF(2)-A/2];

    % isolating cheeks
    leftCheekX = (min(x) + faceWidth*0.3);
    leftCheekY = (min(y) + faceHeight*0.65);
    roiL = [leftCheekX leftCheekY];
    roii{2} = [roiL(1)-A/2 roiL(2)+A/2; roiL(1)-A/2 roiL(2)-A/2; roiL(1)+A/2 roiL(2)+A/2; roiL(1)+A/2 roiL(2)-A/2];

    rightCheekX = (min(x) + faceWidth*0.7);
    rightCheekY = (min(y) + faceHeight*0.65);
    roiR = [rightCheekX rightCheekY];
    roii{3} = [roiR(1)-A/2 roiR(2)+A/2; roiR(1)-A/2 roiR(2)-A/2; roiR(1)+A/2 roiR(2)+A/2; roiR(1)+A/2 roiR(2)-A/2];
    
    bboxes = [roii{1}(1,1) roii{1}(2,2) 2*A A; roii{2}(1,1) roii{2}(2,2) A A; roii{3}(1,1) roii{3}(2,2) A A];
    Ifaces=insertObjectAnnotation(img, 'rectangle', bboxes, 'ROI');
    imagesc(Ifaces), title('Detected forehead'), drawnow;
else
    roii{1} = 1;
    roii{2} = 1;
    roii{3} = 1;
    roiHead = 1;
end
rois = cell(1,2);
rois = {roii, roiHead};
end
'''

# NICU.py functions

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
        
        #using face coordinates, detect cheek and forehead region
        l_eye = (ll[0], ll[5])
        r_eye = (ll[1], ll[6])
        nose = (ll[2], ll[7])
        l_lip = (ll[3], ll[8])
        r_lip = (ll[4], ll[9])
        global roi
        roi = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])

        A = np.sqrt((x2 - x1) * (y2 - y1)) * 0.2 #face scale proportion constant
        l_cheek = (np.average([l_eye[0], l_lip[0]]), np.average([l_eye[1], l_lip[1]])) #left cheek center coordinates
        r_cheek = (np.average([r_eye[0], r_lip[0]]), np.average([r_eye[1], r_lip[1]])) #right cheek center coordinates
        forehead = (np.average([l_eye[0], r_eye[0]]), np.average([l_eye[1], r_eye[1]]) - A) #forehead center coordinates

        #determine if side of face is showing 
        if nose[0] > np.average([l_eye[0], l_lip[0]]):
            cv2.rectangle(img, (int(l_cheek[0] - A/2), int(l_cheek[1] - A/2)), (int(l_cheek[0] + A/2), int(l_cheek[1] + A/2)), CBOX_COLOR, 2)
            roi[0] = np.array([int(l_cheek[0] - A/2), int(l_cheek[1] - A/2), int(l_cheek[0] + A/2), int(l_cheek[1] + A/2)])
        if nose[0] < np.average([r_eye[0], r_lip[0]]):
            cv2.rectangle(img, (int(r_cheek[0] - A/2), int(r_cheek[1] - A/2)), (int(r_cheek[0] + A/2), int(r_cheek[1] + A/2)), CBOX_COLOR, 2)
            roi[1] = np.array([int(r_cheek[0] - A/2), int(r_cheek[1] - A/2), int(r_cheek[0] + A/2), int(r_cheek[1] + A/2)])
        if nose[0] > np.average([l_eye[0], l_lip[0]]) and nose[0] < np.average([r_eye[0], r_lip[0]]):    
            cv2.rectangle(img, (int(forehead[0] - A/2), int(forehead[1] - A/2)), (int(forehead[0] + A/2), int(forehead[1] + A/2)), FBOX_COLOR, 2)
            roi[2] = np.array([int(forehead[0] - A/2), int(forehead[1] - A/2), int(forehead[0] + A/2), int(forehead[1] + A/2)])

    return img,roi


def loop_and_detect(cam, mtcnn, minsize):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    length = 20
    global frame
    frame = 0
    global roii
    roii = np.array([[[0 for k in range(4)] for j in range(3)] for i in range(length)])
    while frame < length+1:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            if len(dets) == 0:
                roii[frame] = roii[frame-1]
            else:
                img,roii[frame] = show_faces(img, dets, landmarks)

            img = show_fps(img, fps)

            #crop ROI and average out pixel intensities
            lroi = img[roii[frame][0][1]:roii[frame][0][3], roii[frame][0][0]:roii[frame][0][2]]
            rroi = img[roii[frame][1][1]:roii[frame][1][3], roii[frame][1][0]:roii[frame][1][2]]
            froi = img[roii[frame][2][1]:roii[frame][2][3], roii[frame][2][0]:roii[frame][2][2]]

            if rroi.any():
                l_roi = np.concatenate(lroi, axis=0)
                r_roi = np.concatenate(rroi, axis=0)
                f_roi = np.concatenate(froi, axis=0)
                avgl = np.mean(l_roi, axis=0)
                avgr = np.mean(r_roi, axis=0)
                avgf = np.mean(f_roi, axis=0)
                avg = sum([avgl,avgr,avgf])
                
                #insert averages into respective rgb array
                r[it][frame] = avg[0]
                g[it][frame] = avg[1]
                b[it][frame] = avg[2]

                print('--------FRAME ' + str(frame) + '--------')

                print('red:' + str(avg[0]))
                print('green:' + str(avg[1]))
                print('blue:' + str(avg[2]))

            
            frame += 1

            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        #key = cv2.waitKey(1)
        #if key == 27:  # ESC key: quit program
        if frame == length:
            return r,g,b,fps
            break
        #elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            #full_scrn = not full_scrn
            #set_display(WINDOW_NAME, full_scrn)


def main():
    HR = np.zeros(loop)
    global it
    it = 0
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    mtcnn = TrtMtcnn()

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT MTCNN Demo for Jetson Nano')

    while it < loop+9:
        r,g,b,fps = loop_and_detect(cam, mtcnn, args.minsize)

        if it >= 9:
            # once 200 frames are collected it starts the HR est pipeline
            rr = np.concatenate(r[it-9:it]) # always uses the 200 most recent frames
            gg = np.concatenate(g[it-9:it])
            bb = np.concatenate(b[it-9:it])
            

            #for i in range(0, length):
            #    if i < sliding_window_size:
            #        continue
            #    r_sliding_window_avg[i] = np.mean(r[i - sliding_window_size:i], axis=0)
            #    b_sliding_window_avg[i] = np.mean(b[i - sliding_window_size:i], axis=0)
            #    g_sliding_window_avg[i] = np.mean(g[i - sliding_window_size:i], axis=0)

            # PPG plot
            #plt.plot(range(0, length), r, 'r')
            #plt.plot(range(0, length), g, 'g')
            #plt.plot(range(0, length), b, 'b')
            #plt.show()

            # detrend data
            X = np.array([rr,gg,bb])
            Xd = detrend(X)

            # normalize data
            #X = Xd.reshape(-1,1)
            Xn = normalize(Xd,axis=0)

            # PCA feature selection of the 3 channels
            pca = PCA(n_components=3)
            pca.fit(Xn)
            #print(pca.explained_variance_ratio_)
            #print(pca.singular_values_)
            #print(Xp)

            Xn  = Xn[2] # green channel was selected based off on PCA results
            fs = fps # detected frames per second
            print(fps)

            # Bandpass filter data
            nyq = 0.5*fs
            low = 0.5 / nyq
            high = 2.5 / nyq
            b,a = butter(6,[low,high],btype='band')
            y = lfilter(b,a,Xn)
            yf = fft(y)
            T = 1 / fs
            N = len(Xn)
            xx = np.linspace(0,1/(2*T),N//2)
            poww = 2.0/N * np.abs(yf[0:N//2])
            #plt.plot(xx,poww)
            #plt.show()

            # peak detector to find frequency of avg HR
            ind = np.where(poww == np.amax(poww))
            HR[it-9] = xx[ind[0]][0]*60
            print('Average HR is: ', HR[it-9])

        it += 1
    cam.stop() 
    cam.release()
    cv2.destroyAllWindows()

    del(mtcnn)


if __name__ == '__main__':
    main()













