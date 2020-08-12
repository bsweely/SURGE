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

Notes:
    1. PIL is consitently 3 to 4 times as slow as cv2 when processing images, according to this site:

    https://www.kaggle.com/vfdev5/pil-vs-opencv

    So, I am using cv2 to so the image processing and loading here instead of Pillow


'''

class PixelBox():
    roi = [[0, 0], [0, 1], [1, 1], [1, 0]]
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
        image = cv2.imread(image) # in BGR format, not RGB

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
    for i in range(start, length + 1, step):
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
    firstImage = cv2.imread(images[0]) # Note: This is in BGR format, not RGB

    # getting a grayscale image for the transform
    firstImageGray = cv2.cvtColor(firstImage, cv2.COLOR_BGR2GRAY)
    # firstImageGray = np.float32(firstImageGray)
    firstImageGray = firstImageGray.astype(np.float32)

    # printing a grayscale color number to make sure that the array is in np.float32 type.
    print("Type of each value in firstImageGray after .astype conversion: ", type(firstImageGray[0, 0]))
    # prints np.float32, so this is working. Why is it not working in the getWarpPerspective function?

    # Getting points to track - There is a GoodFeaturesToTrack method in cv2 online, like in the MATLAB code, so
    # we can try that if the cornerHarris method is not as good
    # Visit this URL for information on GoodFeaturesToTrack in Python
    # https://www.geeksforgeeks.org/python-corner-detection-with-shi-tomasi-corner-detection-method-using-opencv/

    # Testing whether cv2.GoodFeatureToTrack is better then cornerHarris method
    # firstImagePoints = cv2.cornerHarris(firstImageGray, 2, 3, 0.04)
    firstImagePoints = cv2.goodFeaturesToTrack(firstImageGray,25,0.01,10)
    firstImagePoints = firstImagePoints.astype(np.float32)

    # printing a grayscale color number to make sure that the array is in np.float32 type.
    print("Type of each value in firstImagePoints after .astype conversion: ", type(firstImagePoints[0, 0]))
    # prints np.float32, so this is working. Why is it not working in the getWarpPerspective function?

    for image in images:
        if image != images[0]: # if this is the second to Nth image:
            # Converting the images into RGB np arrays from BGR np arrays,
            # assuming that the image outpout from camera.capture_sequence is not RGB
            image = cv2.imread(image)

            # Now tracking image to transform according to the
            # first image in the array

            # image must be in gray scale for corner detection
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image2 = np.float32(image2)
            image2 = image2.astype(np.float32)

            # printing a grayscale color number to make sure that the array is in np.float32 type.
            print("Type of each value in image2 after .astype conversion: ", type(image2[0, 0]))
            # prints np.float32, so this is working. Why is it not working in the getWarpPerspective function?

            # showing image to check conversion
            # plt.show(image2)

            # Testing whether cv2.GoodFeatureToTrack is better then cornerHarris method

            # Getting the corners of the image with Harris Corners Method
            # newImagePoints = cv2.cornerHarris(image2, 2, 3, 0.04)
            newImagePoints = cv2.goodFeaturesToTrack(image2,25,0.01,10)
            newImagePoints = newImagePoints.astype(np.float32)
            print("checking corner objects: ", newImagePoints)

            # printing a grayscale color number to make sure that the array is in np.float32 type.
            print("Type of each value in newImagePoints after .astype conversion: ", type(newImagePoints[0, 0]))
            # prints np.float32, so this is working. Why is it not working in the getWarpPerspective function?

            # Transforming the image in accordance with the first image
            print("type of newImagePoints: ", type(newImagePoints))
            print("printing newImagePoints: ", newImagePoints)
            print("printing the type of each point in newImagePoints: ", type(newImagePoints[0]))
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
    r = np.zeros(60)
    g = np.zeros(60)
    b = np.zeros(60)

    # Iterators
    i = 0
    j = 0
    k = 0

    # files
    images = getListOfJPGs(length = frametotal)

    # Connect to camera
    camera = picamera.PiCamera()
    resolution = (640, 480)
    camera.resolution = resolution
    # camera.brightness = 100 # This brightness level renders the images as white boxes on the screen because the image is so bright
    camera.framerate = framerate

    # Connect to camera
    camera.start_preview()
    time.sleep(2)
    camera.capture_sequence(images, use_video_port = True)
    camera.stop_preview() # added to try to take away the white screen
    '''
    with picamera.PiCamera() as camera:
        resolution = (640, 480)
        camera.resolution = resolution
        # camera.brightness = 100 # This brightness level renders the images as white boxes on the screen because the image is so bright
        camera.framerate = framerate

        print("Images, print 2: ", images)

        # Connect to camera
        camera.start_preview()
        time.sleep(2)
        camera.capture_sequence(images, use_video_port = True)
        camera.stop_preview() # added to try to take away the white screen
    '''


    # make numpy array from stream - not needed yet, but maybe later

    # Start timer
    t_start = time.time()

    # capture Initial Images, however many are in the frame total to get an initial count of images
    # This code is replaced with the with statement above, but it might be necessary later
    # camera.capture_sequence(images, use_video_port = True)

    while 1:
        i += frametotal # starting at the total frame count to start the moving acerage calculation
        # Gets and stores images
        newImages = getListOfJPGs(start = i, length = i+movingAverageIncrement)
        print(newImages)
        camera.resolution = resolution
        # camera.brightness = 100 # This brightness level renders the images as white boxes on the screen because the image is so bright
        camera.framerate = framerate
        # camera.start_preview()
        # time.sleep(2) # to ready the image preview
        camera.capture_sequence(newImages) # camera.capture_sequence(images, use_video_port = True)
        # camera.stop_preview()

        # transforming images
        t1 = time.time()
        newImages = reformatImages(newImages, resolution)
        t2 = time.time()

        # print("Printing images after their transformation: ", images)

        # appending newly collected images to the
        images.append(newImages)

        # checking for timely function execution
        # print("time to reformat images: ", t2 - t1)

        if framenum == 60:
            t_capture = time.time()
            print("capture rate: %.2f" % (framenum/(t_capture-t_start)), "(Hz)")

        # Limits stored images to 60
        if i == 60:
            # camera.stop_preview() # stop showing images on the screen # This line might not be necessary at this point
            i = 0
            j = 0
            for j in range(j, j+movingAverageIncrement):
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
    for i in range(i, i+movingAverageIncrement):
        images.append(rawCapture.array)
        time.sleep(1/60)
        framenum += 1

if __name__ == "__main__":
    main()