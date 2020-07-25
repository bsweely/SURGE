import numpy as np
import picamera
from picamera.array import PiRGBArray
import time
import cv2
import tensorflow
import keras
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,detrend,welch
from scipy.fft import fft
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, FastICA

WINDOW_NAME = 'main'
BBOX_COLOR = (0, 255, 0)  #green
CBOX_COLOR = (0, 0, 255) #red
FBOX_COLOR = (255, 0, 0) #blue

global length
length = 200
global r,b,g
r = np.zeros((length))
g = np.zeros((length))
b = np.zeros((length))

def imageCollector(camera):
    framenum   = 0
    framerate  = 60
    global images
    images = []
    time.sleep(0.1)
    tic = time.time()
    while 1:
        ret, img = camera.read()
        images.append(img)
        framenum += 1
        time.sleep(1/60)
        
    # Stores 60 frames
    #if i == 60:
    #    i = 0
    #    j = 0
    
        print(framenum)
        if framenum == length: #frametotal:
            toc = time.time()
            tim = toc-tic
            fps = framenum/tim
            return fps, images
            break

def imageDetector(camera):
    framenum   = 0
    framerate  = 60
    global images
    images = []
    img = PiRGBArray(camera, size=(640,480))
    camera.capture(img, format="bgr")
    time.sleep(0.1)
    tic = time.time()
    while 1:
        images.append(img.array)
        framenum += 1
        
    # Stores 60 frames
    #if i == 60:
    #    i = 0
    #    j = 0
    
        print(framenum)
        if framenum == length: #frametotal:
            toc = time.time()
            tim = toc-tic
            fps = framenum/tim
            return fps, images
            break

def running_mean(x,N):
    cumsum = np.cumsum(np.insert(x,0,0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    bb = boxes
    ll = landmarks
    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
    
    #using face coordinates, detect cheek and forehead region
    l_eye = ll['left_eye']
    r_eye = ll['right_eye']
    nose = ll['nose']
    l_lip = ll['mouth_left']
    r_lip = ll['mouth_right']
    global roi
    roi = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])

    A = np.sqrt((x2 - x1) * (y2 - y1)) * 0.2 #face scale proportion constant
    l_cheek = (np.average([l_eye[0], l_lip[0]]), np.average([l_eye[1], l_lip[1]])) #left cheek center coordinates
    r_cheek = (np.average([r_eye[0], r_lip[0]]), np.average([r_eye[1], r_lip[1]])) #right cheek center coordinates
    forehead = (np.average([l_eye[0], r_eye[0]]), np.average([l_eye[1], r_eye[1]]) - A) #forehead center coordinates

    #determine if side of face is showing 
    if nose[0] > np.average([l_eye[0], l_lip[0]]):
        roi[0] = np.array([int(l_cheek[0] - A/2), int(l_cheek[1] - A/2), int(l_cheek[0] + A/2), int(l_cheek[1] + A/2)])
    if nose[0] < np.average([r_eye[0], r_lip[0]]):
        roi[1] = np.array([int(r_cheek[0] - A/2), int(r_cheek[1] - A/2), int(r_cheek[0] + A/2), int(r_cheek[1] + A/2)])
    if nose[0] > np.average([l_eye[0], l_lip[0]]) and nose[0] < np.average([r_eye[0], r_lip[0]]):    
        roi[2] = np.array([int(forehead[0] - A/2), int(forehead[1] - A/2), int(forehead[0] + A/2), int(forehead[1] + A/2)])

    return img,roi


def loop_and_detect(imgg, mtcnn):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    global frame
    frame = 0
    global roii
    roii = np.zeros((length,3,4))
    while frame < length:
        img = imgg[frame]
        face = mtcnn.detect_faces(img)
        print(face)
        box = face[0]['box']
        keypoints = face[0]['keypoints']
        #print('{} face(s) found'.format(len(dets)))
        if len(box) == 0:
            roii[frame] = roii[frame-1]
        else:
            img,roii[frame] = show_faces(img, box, keypoints)

        #img = show_fps(img, fps)

        #crop ROI and average out pixel intensities
        lroi = img[int(roii[frame][0][1]):int(roii[frame][0][3]), int(roii[frame][0][0]):int(roii[frame][0][2])]
        rroi = img[int(roii[frame][1][1]):int(roii[frame][1][3]), int(roii[frame][1][0]):int(roii[frame][1][2])]
        froi = img[int(roii[frame][2][1]):int(roii[frame][2][3]), int(roii[frame][2][0]):int(roii[frame][2][2])]

        if rroi.any():
            l_roi = np.concatenate(lroi, axis=0)
            r_roi = np.concatenate(rroi, axis=0)
            f_roi = np.concatenate(froi, axis=0)
            avgl = np.mean(l_roi, axis=0)
            avgr = np.mean(r_roi, axis=0)
            avgf = np.mean(f_roi, axis=0)
            avg = sum([avgl,avgr,avgf])
            
            #insert averages into respective rgb array
            r[frame] = avg[0]
            g[frame] = avg[1]
            b[frame] = avg[2]

            
        frame += 1
        print(frame)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        if frame == length:
            return r,g,b,fps
            break


def main():
    # Variables:
    framerate  = 60

    # Connect to camera
    # camera = picamera.PiCamera()
    # camera.resolution = (640, 480)
    #camera.resolution = (640, 480)
    #camera.brightness = 100
    #camera.framerate = framerate
    
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    
    tic = time.time()
    print("____Running Image Capture_____")
    new_fps, images = imageCollector(camera)
    
    print(new_fps)
    toc = time.time()
    timm = (toc - tic)
    tim = np.linspace(0,timm,num=length)
    
    detector = MTCNN()
    print("____Running Face Detection____")
    r,g,b,fps = loop_and_detect(images, detector)
    print("FPS of system: ", fps)

    X = np.array([r,g,b,tim])

    np.save('/home/pi/Desktop/data/trial1.npy',X)

    # detrend data
    Xd = detrend(X)

    # Consider adding Hamming and Interpolation here

    # normalize data
    Xn = normalize(Xd,axis=0)

    wsize = 5
    Xn = Xn[1] # green channel was selected
    Xnn = running_mean(Xn,wsize)
    plt.plot(range(0,length-(wsize-1)),Xnn,'r')
    plt.plot(range(0,length-(wsize-1)),Xn[(wsize-1):],'g')
    plt.show()

    # Bandpass filter data
    fps = new_fps
    nyq = 0.5*fps
    low = 1 / nyq
    high = 3 / nyq
    b,a = butter(3,[low,high],btype='band')
    y = lfilter(b,a,Xnn)
    yf = fft(y)
    T = 1 / fps
    N = len(Xn)
    xx = np.linspace(0,1/(2*T),N//2)
    poww = 2.0/N * np.abs(yf[0:N//2])

    # peak detector to find frequency of avg HR
    ind = np.where(poww == np.amax(poww))
    HR = xx[ind[0]][0]*60
    print('Average HR is: ', HR)

    plt.plot(tim[(wsize-1):],y,'g')
    plt.show()
    plt.plot(xx,poww)
    plt.show()

if __name__ == "__main__":
    main()
