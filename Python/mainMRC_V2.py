from io import BytesIO
import time
import numpy
import picamera
import picamera.array
import io
import scipy
import scipy.signal
import imageio
import cv2

'''
Herein are the classes and functions that we need to execute the MRC algorithm.

Things not implemented yet include:
    1. Goodness metric coarse estimation
    
    2. Face tracking to transform new images to fit the previous ones
    
Ideas:
    1. Have the face detection detect faces like before, and when face tracking
    is implemented, then use the same roi coordinates as before to reduce computational
    load. 
    
    2. Perhaps define the quantity of new images for the moving average and
    make the code transform each image and use the same roi coordinates for
    this defined set of images.
    
    3. After each iteration of the gathering new images, the roi coordinates, 
    face tracking points, and the chosen best region of interest are all reset
    and recalculated to account for movement of the patient.
    
    4. Have a feature of the face detection to detect of the face is in a profile
    position, as newborns often sleep with their heads to the side. 

How this program works
    1. This program takes a series of images quickly and processes them offline,
    for the Raspberry Pi doesnt seem able to process the images quickly enough
    online
'''

class PixelBox():
    roi = [[0, 0], [0, 1], [1, 1], [1, 0]]
    rIntensity = 0
    bIntensity = 0
    gIntensity = 0
    

def get20x20PixelRegions(image, minX, maxX, minY, maxY):
    '''
    

    Parameters
    ----------
    image : np.array
        DESCRIPTION.
    minX : minimumm X coordinate in the roi of the image
        DESCRIPTION.
    maxX : maximum X coordinate in the roi of the image
        DESCRIPTION.
    minY : minimumm Y coordinate in the roi of the image
        DESCRIPTION.
    maxY : maximum Y coordinate in the roi of the image
        DESCRIPTION.

    Returns
    -------
    listOf20x20Regions : Includes the 20x20 pixel regions

    '''
    xPoints = np.arange(minX, maxX, 20)
    yPoints = np.arange(minY. maxY, 20)
    listOf20x20Regions = []
    index = 0
    
    for xPoint in range(len(xPoints)):
        for yPoint in range(len(yPoints)):
            roi = [[]]
    
    return listOf20x20Regions

def getMaxAndMinXAndY(bbox):
    '''
    Parameters
    ----------
    bbox : [x, y, w, h] like in MATLAB

    Returns
    -------
    list
        [minX, maxX, minY, maxY]

    '''
    (x, y, w, h) = bbox
    
    return [x, x+w, y, y+h]

def bbox2points(bbox):
    '''
    This function returns a set of four points from a bbox, which is the
    format of the faces from the detectfaces function, like how it was in MATLAB.
    
    Not used yet becasue the ROI in this program is not structured in this way,
    and we do not need this function so far.
    '''
    (x, y, w, h) = bbox
    roi = np.array([[x, y], [x, y+h], [x+w, y], [x+w, y+h]])
    
    return roi
    
def getBiggestDetectedFace(faces):
    '''
    This function intakes the faces array that is made from the cv2 face
    classifier, so it decides which face is mostlikely the true one by finding
    the biggest detected face in the list of faces
    '''
    faceAreas = []
    for (x, y, w, h) in faces:
        faceAreas.append(w*h)
    index = faceAreas.index(max(faceAreas))
    
    # returning largest face, probably the true face incase of errors in face detection
    return faces[index]
        

def reduceImagesToFaces(images):
    '''
    This is the analog to detectfaces.m in the MATLAB code. The code here
    is sourced from a github, which I have downloaded. This function intakes
    the array of images and returns an array of the same images but with
    the detected faces in them.
    
    This does not yet detect faces every 5 frames or so. It detects the face
    in each image. 
    '''
    cascadePath = 'haarcascade_frontalface_default.xml'
    faceDetector = cv2.CascadeClassifier(cascadePath)
    for image in images:
        image = cv2.imread(image)
        
        # may not need this conversion if we read the images with the imageio
        # method, for it makes a numpy array of RGB image data
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detecting faces
        faces = faceCascade.detectMultiScale(
            grayimage,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
            )
        print("{0} faces were found for image: %02d".format(len(faces)) % image)
        
        if len(faces) != 0 or images.index(image) == 1: # testing to make sure that there is indeed a face
            for (x, y, w, h) in faces: # notice that the format (x, y, w, h) is the bbox format
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("detect face(s)", image)
            cv2.waitKey(0)
            
            biggestFace = getBiggestDetectedFace(faces)
            [minX, maxX, minY, maxY] = getMaxAndMinXAndY(biggestFace)
            
            image = image[minX:maxX, minY:maxY]
            
    return images 
        
def getListOfJPGs(length, start = 1, step = 1):
    images = []
    for i in range(start, length, step):
        images.append('image%02d.jpg' % i)
    return images

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

def reformatImages(images, resolution):
    '''

    Parameters
    ----------
    images : list of images

    Returns
    -------
    transformedImages : list of images that are transformed
    according to the first image
    
    This function is for transforming the images into RGB data, finding 
    corners to track, and then repositioning the images after image 1 to fit 
    image 1.

    '''
    # Getting the first image for reference
    firstImage = imagesio.imread(images[0])
    firstImageGray = np.copy(firstImage)
    firstImageGray = RGb2gray(firstImageGray)
    
    firstImageCorners = cv2.cornerHarris(firstImageGray, 4, 3, 0.04)
    firstImagePoints = np.float32(firstImageCorners)
    for image in images:
        if image != images[0]: # if this is the second to Nth image:
            # Converting the images into RGB numpy arrays from BGR numpy arrays,
            # assuming that the image outpout from camera.capture_sequence is not RGB
            image = imagesio.imread(image)
            
            # Now tracking image to transform according to the 
            # first image in the array
            # making a copy of the image into RGB form
            image2 = np.copy(image)
            
            # image must be in gray scale for corner detection
            image2 = RGB2gray(image2)
            
            # showing image to check conversion
            plt.show(image2)
            
            # Getting the corners of the image with Harris Corners Method
            corners = cv2.cornerHarris(image2, 4, 3, 0.04)
            print("checking corner objects: ", corners)
            
            # Transforming the image in accordance with the first image
            newImagePoints = np.float32(corners)
            transform = cv2.getPerspectiveTransform(firstImagePoints, newImagePoints)
            image = cv2.warpPerspective(image, transform, (image.shape[1], images.shape[0]), flags = cv2.INTER_LINEAR)
        else:
            pass # if this is the first image
    
    return images
        

def main():
    # Variables:
    framenum   = 0
    framerate  = 120
    frametotal = 60
    movingAverageIncrement = 10

    images = []
    r = numpy.zeros(60)
    g = numpy.zeros(60)
    b = numpy.zeros(60)

    # Iterators
    i = 0
    j = 0
    k = 0

    # files
    images = getListOfJPGs(length = frametotal)
    # print(images)

    # Connect to camera
    camera = picamera.PiCamera()
    resolution = (640, 480)
    camera.resolution = resolution
    # camera.start_preview() # inserted. Perhaps not needed.
    # time.sleep(2)
    camera.brightness = 100
    camera.framerate = framerate
    # camera.start_preview()
    time.sleep(5)

    # Start timer
    t_start = time.time()

    # capture Initial Images, however many are in the frame total to get an initial count of images
    camera.capture_sequence(images, use_video_port = True)

    while 1:
        i+=1
        # Gets and stores images
        images = getListOfJPGs(start = i, length = i+movingAverageIncrement)
        # print(images)
        camera.capture_sequence(images, use_video_port = True)
        
        # transforming images
        t1 = time.time()
        images = reformatImages(images, resolution)
        t2 = time.time()
        
        # checking for timely function execution
        print("time to reformat images: ", t2 - t1)
        
        if framenum == 60:
            t_capture = time.time()
            print("capture rate: %.2f" % (framenum/(t_capture-t_start)), "(Hz)")

        # Limits stored images to 60
        if i == 60:
            camera.stop_preview() # stop showing images on the screen
            i = 0
            j = 0
            for j in range(j, j+10):
                # Find ROI and crop array

                # Get RGB intensities
                print(j)
                print('j is %02d' % j)
                print(images[j])
                print(images[j][1])
                r[j] = numpy.sum(images[j][:][:][2])
                g[j] = numpy.sum(images[j][:][:][1])
                b[j] = numpy.sum(images[j][:][:][0])

                # Detrend RGB intensities
                r_detrend = scipy.signal.detrend(r)
                g_detrend = scipy.signal.detrend(g)
                b_detrend = scipy.signal.detrend(b)

                # Normalize RGB intensities
                # z = (x-mu)/sigma
                r_norm = (r_detrend - r_detrend.mean(0)) / r_detrend.std(0)
                g_norm = (g_detrend - r_detrend.mean(0)) / r_detrend.std(0)
                b_norm = (b_detrend - r_detrend.mean(0)) / r_detrend.std(0)

        if framenum >= 60: #frametotal:
            camera.close()
            t_finish = time.time()
            t_total = t_finish-t_start
            print("time to run : %.3f" % (t_total), "(s)")
            #print()
            #print(r_norm)
            #print()
            #print(g_norm)
            break


    ### TESTING ###  ### TESTING ###

    ### TESTING ###  ### TESTING ###

### Functions ###

def GetImage(images, i, framenum, rawCapture):
    print("yes")
    for i in range(i, i+movingAverageIncrement:
        images.append(rawCapture.array)
        time.sleep(1/60)
        framenum += 1

if __name__ == "__main__":
    main()