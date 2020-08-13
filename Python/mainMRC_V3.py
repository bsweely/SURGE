from io import BytesIO
import time
import numpy as np
import picamera
import picamera.array
import io
import scipy
import scipy.signal
import imageio
import cv2
import matplotlib.pyplot as plt

'''
Herein are the classes and functions that we need to execute the MRC algorithm.

Things not implemented yet include:
    1. Goodness metric coarse estimation

    2. Face tracking to transform new images to fit the previous ones

Ideas:
    1. Have the face detection detect faces like before, and when face tracking
    is implemented, then use the same roi coordinates as before to reduce computational
    load.

    2. After each iteration of the gathering new images, the roi coordinates,
    face tracking points, and the chosen best region of interest are all reset
    and recalculated to account for movement of the patient.

    3. Have a feature of the face detection to detect of the face is in a profile
    position, as newborns often sleep with their heads to the side.

Steps for Image Processing:

    1. Collect a list of images, where each frame has at least one face in it -> getImagesInformation()

    2. Transform the images to have matching coordinates for the faces, and isolate the faces from the images -> reformatImages()

    3. Extract the RGB data from each set of images -<> getRGBFromImages()

    When the MRC algorithm is fully implemented:

    4. divide the face into 20x20 pixel regions

    5. use Goodness metric to choose certain rois for the remainder of some time before resetting to new rois

Notes:
    1. PIL is consitently 3 to 4 times as slow as cv2 when processing images, according to this site:

    https://www.kaggle.com/vfdev5/pil-vs-opencv

    So, I am using cv2 to so the image processing and loading here instead of PIL


'''

class FrameArray():
    framesList = np.array([])
    fps = 0
    faceCorners = np.array([])
    firstFaceXY = np.array([])

    def __init__(self, camera, length, start = 1, step = 1, showImages = False):

        '''
        This function captures the images for each increment in the moving window of heart rates,
        with the option of showing the images or not.

        This is not finished yet.
        '''

        # Looping to collect a set of images until a face is detected in each frame
        # If there is a missing face in the current image, then there is a new image capture taken.
        # If there is at least one face in each frame, then this function completes. If one frame has no detected faces,
        # the the function continues and faceInEachFrame = false, thus repeating the loop and collecting new images
        faceInEachFrame = False

        while faceInEachFrame == False:
            faceInEachFrame = True
            images = []
            fps = 0
            for i in range(start, length + 1, step):
                images.append('image%02d' % i)

            if showImages == False:
                t_start = time.time()
                camera.capture_sequence(images, format = 'BGR')
                t_capture = time.time()
            else:
                camera.start_preview()
                time.sleep(2)
                t_start = time.time()
                camera.capture_sequence(images, format = 'BGR')
                t_capture = time.time()
                camera.stop_preview()

            # calculating frames per second for image capture
            fps = length/(t_capture - t_start)

            # checking that there is a face in each image before accepting the
            # list of images for heart rate detection
            cascadePath = 'haarcascade_frontalface_default.xml'
            faceDetector = cv2.CascadeClassifier(cascadePath)
            faceCorners = np.array([])
            faces = np.zeros(len(images))
            print("debugging: length of faces before initializing with faces: ", length(faces))

            # iterating through each index in the images array to detect a face in the image
            image = 0

            while image < range(length(images)): # iterating through each index of the images array to detect faces
                images[image] = cv2.imread(images[images]) # in BGR format, not RGB

                grayImage = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)

                # Detecting faces
                face = faceCascade.detectMultiScale(
                    grayImage,
                    scaleFactor = 1.1,
                    minNeighbors = 5,
                    minSize = (30, 30)
                    #flags = cv2.CV_HAAR_SCALE_IMAGE
                    )

                # Testing whether there is a face in this frame. If not, then restart the image collection
                if len(faces) == 0:
                    faceInEachFrame = False
                    break

                faces.append(face)

                # Getting biggest face
                for (x, y, w, h) in faces[image]: # notice that the format (x, y, w, h) is the bbox format
                    cv2.rectangle(images[image], (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("detect face(s)", images[image])
                cv2.waitKey(0)

                biggestFace = getBiggestDetectedFace(faces[image])
                [minX, maxX, minY, maxY] = getMaxAndMinXAndY(biggestFace)
                if image == 0:
                    firstFaceXY = (minX, maxX, minY, maxY)
                faceCorners[image] = np.array([[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]])
                # images[image] = images[image][minX:maxX, minY:maxY]

        self.framesList = images
        self.fps = fps
        self.faceCorners = faceCorners
        self.firstFaceXY = firstFaceXY




class Frame():
    '''
    class used to iterate through pixel boxes.

    This is not used yet because we aren't implementing pixel boxes yet
    '''
    listOfPixelBoxes = np.array([])



class PixelArea():
    roi = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    rIntensity = 0
    bIntensity = 0
    gIntensity = 0

def get20x20PixelRegions(image, minX, maxX, minY, maxY):
    '''
    This function is not finished yet

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
    listOf20x20Regions = np.array([])
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

    return np.array([x, x+w, y, y+h])

def bbox2points(bbox):
    '''
    This function returns a set of four points from a bbox, which is the
    format of the faces from the detectfaces function, like how it was in MATLAB.

    Not used yet because the ROI in this program is not structured in this way,
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
    faceAreas = np.array([])
    for (x, y, w, h) in faces:
        faceAreas.append(w*h)
    index = faceAreas.index(max(faceAreas))

    # returning largest face, probably the true face incase of errors in face detection
    return faces[index]

def getImagesInformation(camera, length, start = 1, step = 1, showImages = False):
    '''
    This function captures the images for each increment in the moving window of heart rates,
    with the option of showing the images or not
    '''

    # Looping to collect a set of images until a face is detected in each frame
    # If there is a missing face in the current image, then there is a new image capture taken.
    # If there is at least one face in each frame, then this function completes. If one frame has no detected faces,
    # the the function continues and faceInEachFrame = false, thus repeating the loop and collecting new images
    faceInEachFrame = False

    while faceInEachFrame == False:
        faceInEachFrame = True
        images = []
        fps = 0
        for i in range(start, length + 1, step):
            images.append('image%02d' % i)

        print("Finished making the list of images")
        print("now taking the pictures...")

        if showImages == False:
            t_start = time.time()
            camera.capture_sequence(images, format = 'BGR')
            t_capture = time.time()
        else:
            camera.start_preview()
            time.sleep(2)
            t_start = time.time()
            camera.capture_sequence(images, format = 'BGR')
            t_capture = time.time()
            camera.stop_preview()

        print("Finished taking the pictures")

        # calculating frames per second for image capture
        fps = length/(t_capture - t_start)

        # checking that there is a face in each image before accepting the
        # list of images for heart rate detection
        cascadePath = 'haarcascade_frontalface_default.xml'
        faceDetector = cv2.CascadeClassifier(cascadePath)
        faceCorners = np.array([])
        faces = np.zeros(len(images))
        print("Finished making the face detector")
        print("debugging: length of faces before initializing with faces: ", length(faces))

        # iterating through each index in the images array to detect a face in the image
        image = 0

        print("Starting the face detection")
        while image < range(length(images)): # iterating through each index of the images array to detect faces
            images[image] = cv2.imread(images[images]) # in BGR format, not RGB

            grayImage = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)

            # Detecting faces
            face = faceCascade.detectMultiScale(
                grayImage,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
                )

            # Testing whether there is a face in this frame. If not, then restart the image collection
            if len(faces) == 0:
                faceInEachFrame = False
                break

            faces.append(face)

            # Getting biggest face
            for (x, y, w, h) in faces[image]: # notice that the format (x, y, w, h) is the bbox format
                cv2.rectangle(images[image], (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("detect face(s)", images[image])
            cv2.waitKey(0)

            biggestFace = getBiggestDetectedFace(faces[image])
            [minX, maxX, minY, maxY] = getMaxAndMinXAndY(biggestFace)
            if image == 0:
                firstFaceXY = (minX, maxX, minY, maxY)
            faceCorners[image] = np.array([[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]])
            # images[image] = images[image][minX:maxX, minY:maxY]

    return (images, fps, faceCorners, firstfaceXY)

def reduceToLastNIndices(array, n):
    length = len(array)
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

def reformatImages(images, faceCorners, firstFaceXY):
    '''
    Paramaters
    ----------
    images : list of images

    Returns
    -------
    images : list of images that are transformed
    according to the first image
    '''

    # collecting initial time for reformatting images
    t_start = time.time()

    # getting a grayscale image for the transform
    firstImageGray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    firstImagePoints = faceCorners[0]

    # Getting X and Y values for first image's face for transformation
    (firstMinX, firstMaxX, firstMinY, firstMaxY) = firstFaceXY

    # iterating through each index in the images array
    for image in range(start = 1, stop = length(images)):
        # Converting the images into RGB np arrays from BGR np arrays,
        # assuming that the image outpout from camera.capture_sequence is not RGB
        newImagePoints = faceCorners[image]

        # image must be in gray scale for corner detection
        image2 = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)

        transform = cv2.getPerspectiveTransform(newImagePoints, firstImagePoints)
        images[image] = cv2.warpPerspective(images[image], transform, (images[image].shape[1], images[image].shape[0]), flags = cv2.INTER_LINEAR)
        images[image] = images[image][firstMinX:firstMaxX, firstMinY:firstMaxY]

    # Getting the final time for reformatting
    t_final = time.time()

    timeElspased = length(images)/(t_finish - t_start)

    return (images, timeElapsed)

def getRGBFromImages(images):

    # Image Data
    r_norm = np.zeros(len(images))
    g_norm = np.zeros(len(images))
    b_norm - np.zeros(len(images))

    for j in range(len(images)):
        # Find ROI and crop array

        # Get RGB intensities
        # In a BGR Numpy array, in the third axis, 0 is B, 1 is G, and 2 is R. 3 is Transparency

        print(j)
        print('j is %02d' % j)
        print(images[j])
        print(images[j][1])
        r[j] = np.sum(images[j][:][:][2])
        g[j] = np.sum(images[j][:][:][1])
        b[j] = np.sum(images[j][:][:][0])

        # Detrend RGB intensities
        r_detrend = scipy.signal.detrend(r)
        g_detrend = scipy.signal.detrend(g)
        b_detrend = scipy.signal.detrend(b)

        # Normalize RGB intensities
        # z = (x-mu)/sigma
        r_norm[j] = (r_detrend - r_detrend.mean(0)) / r_detrend.std(0)
        g_norm[j] = (g_detrend - r_detrend.mean(0)) / r_detrend.std(0)
        b_norm[j] = (b_detrend - r_detrend.mean(0)) / r_detrend.std(0)

        return(r_norm, g_norm, b_norm)

def main():
    # Variables:
    framenum   = 0
    framerate  = 90
    frametotal = 60
    movingAverageIncrement = 10
    images = []
    r = np.zeros(60)
    g = np.zeros(60)
    b = np.zeros(60)

    # Iterators
    i = 0
    j = 0
    k = 0

    # Connect to camera
    camera = picamera.PiCamera()
    resolution = (640, 480)
    camera.resolution = resolution
    # camera.brightness = 100 # This brightness level renders the images as white boxes on the screen because the image is so bright
    camera.framerate = framerate

    # getting initial images, corners, fps, and firstfaceXY
    (initialImages, initialCorners, initialFPS, initialFirstImageXY) = getImagesInformation(camera,
                                                                                            length = frametotal,
                                                                                            showImages = True)
    # checking the FPS for the Initial Images
    print("FPS for initial images: ", initialFPS)

    # Transforming Images
    (initialImages, timeElapsed) = reformatImages(initialImages)

    # Checking the time elapsed for reformatting the initial images
    print("Time elapsed for the initial images: ", timeElapsed)

    while 1:

        # Collecting Images from Camera
        t_start = time.time()
        (newImages, newCorners, newFPS, newFirstImageXY) = getImages(camera,
                                                                     start = i,
                                                                     length = i+movingAverageIncrement,
                                                                     showImages = True)
        # Checking the FPS for the new images
        print("FPS for new images: ", newFPS)

        # Transforming Images
        (newImages, timeElapsed) = reformatImages(newImages)

        # appending newly collected items to original arrays
        images = initialImages + newImages
        images = reduceToLastNIndices(images, frametotal)

        faceCorners = initialCorners + newCorners
        faceCorners = reduceToLastNIndices(faceCorners, frametotal)

        # Getting BGR data from all images

        (r_norm, g_norm, b_norm) = getRGBFromImages(images)

        '''
        # Deallocating unnecesssary image data
        # I might not have to code this in
        initialImages = None
        newImages = None
        initialCorners = None
        newCorners = None
        '''



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
    for i in range(i, i+movingAverageIncrement):
        images.append(rawCapture.array)
        time.sleep(1/60)
        framenum += 1

if __name__ == "__main__":
    main()