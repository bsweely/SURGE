import argparse
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
import os
import shutil
import ffmpy
import subprocess

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

    4. Making the code check for faces while it splits frames is much too slow,
    so if we could paralellize the code to continuously detect a face in the frame
    while it collects a video, then we could take each frame and use it in good faith
    that there is a face in each frame. This might be a lot faster.

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

class FFMPEGFrames:
    # this class depends on the modules os and subprocess
    def __init__(self, output):
        self.output = output

    def extract_frames(self, input, fps):
        output = input.split('/')[-1].split('.')[0]

        if not os.path.exists(self.output + output):
            os.makedirs(self.output + output)

        query = "ffmpeg -i " + input + " -framerate fps=" + str(fps) + " " + self.output + output + "/output%06d.png"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        s = str(response).encode('utf-8')

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
    # print("This is the bbox to be transformed: ", bbox)
    [x, y, w, h] = bbox

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
    the biggest detected face in the list of faces.append

    numpy arrays do not seem to loop with the "in" keyword correctly in Python.
    '''
    faceAreas = np.array([])
    numOfFaces = len(faces)
    for face in np.arange(numOfFaces):
        w = faces[face][2] # width of face
        h = faces[face][3] # height of face
        area = w*h
        if face == 0:
            maxArea = area
            biggestFace = faces[face]
        else:
            if area > maxArea:
                biggestFace = faces[face]

#        faceAreas = np.append(faceAreas, np.array([w*h]))
#    indexOfMaxArea = np.where(faceAreas == max(faceAreas))
#    print("This is the indexOfMaxArea: ", indexOfMaxArea)

    # returning largest face, probably the true face incase of errors in face detection
    # print("printing the faces array: ", faces)
    # print("printing the faces[indexOfMaxArea]: ", faces[[indexOfMaxArea]])
    # print("Size of the faces array: ", faces.shape)
    # biggestFace = faces[indexOfMaxArea]
    # print("Size of biggest face: ", biggestFace.shape)
    return biggestFace

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
    firstImagePoints = np.float32(faceCorners[0])

    # Getting X and Y values for first image's face for transformation
    (firstMinX, firstMaxX, firstMinY, firstMaxY) = firstFaceXY

    # iterating through each index in the images array
    for image in np.arange(start = 1, stop = len(images)):
        # Converting the images into RGB np arrays from BGR np arrays,
        # assuming that the image outpout from camera.capture_sequence is not RGB
        newImagePoints = np.float32(faceCorners[image])

        # image must be in gray scale for corner detection
        image2 = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)

        transform = cv2.getPerspectiveTransform(newImagePoints, firstImagePoints)
        images[image] = cv2.warpPerspective(images[image], transform, (images[image].shape[1], images[image].shape[0]), flags = cv2.INTER_LINEAR)
        images[image] = images[image][firstMinX:firstMaxX, firstMinY:firstMaxY]

    # Getting the final time for reformatting
    t_final = time.time()

    timeElapsed = len(images)/(t_final - t_start)

    return (images, timeElapsed)

def getRGBFromImages(images):

    # Image Data
    r = np.zeros(len(images))
    g = np.zeros(len(images))
    b = np.zeros(len(images))

    for j in np.arange(len(images)):
        # Find ROI and crop array

        # Get RGB intensities
        # In a BGR Numpy array, in the third axis, 0 is B, 1 is G, and 2 is R. 3 is Transparency

        '''
        print(j)
        print('j is %02d' % j)
        print(images[j])
        print(images[j][1])
        '''
        r[j] = np.sum(images[j][:][:][2])
        g[j] = np.sum(images[j][:][:][1])
        b[j] = np.sum(images[j][:][:][0])

    # Detrend RGB intensities
    r_detrend = scipy.signal.detrend(r)
    g_detrend = scipy.signal.detrend(g)
    b_detrend = scipy.signal.detrend(b)

    j = 0
    rMean = r_detrend.mean(0)
    bMean = b_detrend.mean(0)
    gMean = g_detrend.mean(0)

    rSTD = r_detrend.std(0)
    bSTD = b_detrend.std(0)
    gSTD = g_detrend.std(0)

    for j in np.arange(len(images)):
        # Normalize RGB intensities
        # z = (x-mu)/sigma
        r[j] = (r_detrend[j] - rMean) / rSTD
        g[j] = (g_detrend[j] - gMean) / gSTD
        b[j] = (b_detrend[j] - bMean) / bSTD

    return(r, g, b)
def getImagesFromImagesList(images):
    '''
    This function converts the images from capture_sequence
    into actual BGR images, for capture_sequence does not do this on its own.

    The list actualImages is this array of bgr images.
    '''

    actualImages = []
    for image in range(len(images)):
        actualImages.append(cv2.imread(images[image]))

    # print("testing whether the getImagesFromImagesList worked: ", type(actualImages[0]))

    return actualImages

def captureVideoToImages(camera, frametotal, framerate):
    # constants

    # This constant is to add more seconds to the
    # video capture so that enough time is allotted
    # to collect the number of frames specified in frametotal
    TIME_TO_ACCOUNT_FOR_ERROR = 10/framerate;

    # Making directories for the video
    muDirectory = '/home/pi/mu_code'
    videoFilesPath =  muDirectory + '/video_files_MRC_v4'
    print("This is the videoFilesPath: ", videoFilesPath)
    print("This is the current working directory: ", os.getcwd())

    # making the face detector to make sure
    # that there is a face in each frame before proceeding
    # to extract more frames in the video
    cascadePath = "/home/pi/Desktop/SURGE-Project/Python/haarcascade_frontalface_default.xml"
    faceDetector = cv2.CascadeClassifier(cascadePath)
    faceCorners = np.array([])
    faces = np.array([])
    print("Finished making the face detector")
    print("debugging: length of faces before initializing with faces: ", len(faces))


    # making variable that will determing whether we stop getting new videos or not
    mustRecaptureVideoAndFrames = True

    while mustRecaptureVideoAndFrames == True:
        # notes
        '''
        If the variable mustRecaptureVideoAndFrames == False, then this loop
        will loop until the video is processed successfully, or when the video
        can produce all of its frames correctly.

        See if I can initialize camera with a wait time (try camera.open
        or camera.read) for real time capture. Captures one frame at a time.
        Instantaneous.

        Look into avi video format potentially (it is in RGB format).For both real time and offline analysis.
        '''

        # Making directories for files
        try:
            os.mkdir(videoFilesPath)
        except:
            print("The videoFilesPath already exists here.")
            print("making new videoFilesPath.")
            shutil.rmtree(videoFilesPath)
            os.mkdir(videoFilesPath)

        os.chdir(videoFilesPath)

        # making folder for frames
        framesFolderPath = videoFilesPath + '/frames'
        os.mkdir(framesFolderPath)

        # making video file
        videoFileName = 'video.h264'
        mpegVideoFileName = 'video.mp4'
        videoDirectory = videoFilesPath + '/' + videoFileName
        videoDirectory2 = videoFilesPath + '/' + mpegVideoFileName
        timeToCaptureVideo = frametotal/framerate # + TIME_TO_ACCOUNT_FOR_ERROR

        print("Time to record video for this iteration: ", timeToCaptureVideo)
        time.sleep(2)

        # Capturing Video
        print("Starting video capture")
        camera.start_recording(videoDirectory, format = 'h264')
        tic = time.time()
        print("finished video capture")
        camera.wait_recording(timeToCaptureVideo)
        toc = time.time()
        camera.stop_recording()
        timeCapture = toc - tic
        print("Time to capture video is: ", timeCapture)

        # The Pi is having trouble taking video with mp4
        # So, here, we convert it to mp4 with ffmpeg
        ff = ffmpy.FFmpeg(inputs={videoFileName: None}, outputs={mpegVideoFileName: None})
        ff.run()

        # Here, we are extracting frames with FFmpeg
        f = FFMPEGFrames(framesFolderPath + "/")
        f.extract_frames(videoDirectory2, framerate)
        print("Sleeping for 20 seconds")
        time.sleep(20)

        # construct the video object'=
        print("Printing videoDirectory string: ")
        print(videoDirectory2)
        video = cv2.VideoCapture(videoDirectory2)
        frames = np.array([])

        # checking to make sure that all of the frames were captured
        imageNumber = 1

        # initially assuming frames are correctly captured
        correctlyCapturedAFrame = True

        # Making sure that all frames are present in the video that was captured
        while correctlyCapturedAFrame == True:
            if imageNumber == frametotal + 1:
                # break from the loop, for all
                # desired images are collected
                break

            print("Frame: ", imageNumber)
            correctlyCapturedAFrame, frame = video.read()

            if correctlyCapturedAFrame == False:
                # There is a frame error - a frame that is not correctly
                # captured from video.read()
                print("\nThere was a frame that was not extracted correctly.")
                print("Frame not extracted correctly: ", imageNumber)
                print("Now capturing a new video and deleting old video files.\n")

                '''
                # change to the original directory
                os.chdir(muDirectory)

                # delete file with both video and image files in it
                shutil.rmtree(videoFilesPath)
                mustRecaptureVideoAndFrames = True

                # resetting loop variables
                imageNumber = 0
                '''

                # Restarting loop for capturing video
                break

            else:
                # If there is a frame correctly captured from the video files

                # Making sure that there is a face in each image
                # Before proceeding to extract more images
                # Getting and checking grayscale version of the image
                # print("Checking image shape before processing: ", frame.shape)

                # Commenting out the face detection stuff to test
                # the speed of image analysis

                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # print("Checking gray image shape before processing: ", grayImage.shape)

                # Detecting facces
                facesDetected = faceDetector.detectMultiScale(
                    grayImage,
                    scaleFactor = 1.1,
                    minNeighbors = 5,
                    minSize = (30, 30))
                    # flags = cv2.CV_HAAR_SCALE_IMAGE)

                # If a frame does not have any detected faces
                if len(facesDetected) == 0:
                    print("Frame number", imageNumber,"does not have a face in it. Resetting video and images now")
                    correctlyCapturedAFrame = False
                    mustRecaptureVideoAndFrames = True

                    # Restarting loop for capturing video
                    break

                else:
                    # If there is at least one face, then get biggest face
                    print("Capturing a frame value: ", correctlyCapturedAFrame)
                    numOfFaces = len(facesDetected)
                    for values in np.arange(numOfFaces): # notice that the format (x, y, w, h) is the bbox format
                        # when I use the "in" keyword, the numpy array is not iterated through correctly
                        # in this loop
                        x = facesDetected[values][0]
                        y = facesDetected[values][1]
                        w = facesDetected[values][2]
                        h = facesDetected[values][3]
                        # print("Current bbox being printed: ", facesDetected[values])


                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("detect face(s)", frame)
                    # cv2.waitKey(0)

                    biggestFace = getBiggestDetectedFace(facesDetected)
                    [minX, maxX, minY, maxY] = getMaxAndMinXAndY(biggestFace)

                    if imageNumber == 1:
                        firstFaceXY = (minX, maxX, minY, maxY)
                    faceCorners = np.append(faceCorners, (np.array([[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]])))
                    # print("printing faceCorners in the getImagesInformationFunction before transformation: ", faceCorners)
                    # frame = frame[minX:maxX, minY:maxY]

                    # reshaping faceCorners to have separate faceCorners instead
                    # of having all of them in one dimension.
                    numValuesInFaceCoords = len(faceCorners)
                    NUM_OF_COORDS_INSTANCES = numValuesInFaceCoords/8 # each face has 4
                    NUM_OF_COORDS_INSTANCES = int(NUM_OF_COORDS_INSTANCES)
                    # print("number of face coordinates present: ", NUM_OF_COORDS_INSTANCES)
                    faceCorners = np.reshape(faceCorners, (NUM_OF_COORDS_INSTANCES, 4, 2))
                    # print("printing faceCorners in the getImagesInformationFunction after transformation: ", faceCorners)
                    # print("Image number we are on: ", imageNumber)
                    cv2.imwrite("frame%d.jpg" % imageNumber, frame)


                # Now, the cv2.waitkey() function is not working
                '''
                if cv2.waitkey(10) == 27: # if someone hits the escape key
                    print("The escape key was hit, so this video and its files will stop being processed")
                    break
                '''

                imageNumber += 1

            # Releasing the video reader and all headers
            video.release()
            cv2.destroyAllWindows()

        if frametotal == imageNumber:
            # making list of image files
            frames = os.listdir(videoFilesPath)
            print("Type of object in frames list: ", type(frames[0]))
            print("printing the frames directory, sorted: ", np.sort(frames))
            break

        # If mustRecaptureVideoAndFrames == False, then the loop will loop again
    frames = getImagesFromImagesList(frames)
    fps = framerate
    return (frames, fps, faceCorners, firstFaceXY, timeCapture)




def main():
    # Variables:
    framenum   = 0
    framerate  = 60 # Changing the framerate (as an experiment) did not change fps
    frametotal = framerate*30
    movingAverageIncrement = 10
    images = []
    r = np.zeros(frametotal)
    g = np.zeros(frametotal)
    b = np.zeros(frametotal)

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
    (initialImages, initialFPS, initialCorners, initialFirstImageXY, timeElapsed) = captureVideoToImages(camera, frametotal, framerate)
    # checking the FPS for the Initial Images
    print("FPS for initial images: ", initialFPS)
    # print("face corners for initial images: ", initialCorners)
    # print("elment one and its type of initialCorners: ", initialCorners[0], type(initialCorners[0]))

    # Transforming Images
    (initialImages, timeElapsed) = reformatImages(initialImages, initialCorners, initialFirstImageXY)

    # Checking the time elapsed for reformatting the initial images
    # print("Time elapsed for the initial images: ", timeElapsed)

    while 1:

        # Collecting Images from Camera
        t_start = time.time()
        (newImages, newFps, newCorners, newFirstImageXY, newTimeElapsed) = getImagesInformation(camera,
                                                                                frametotal,
                                                                                framerate)
        # Checking the FPS for the new images
        print("FPS for new images: ", newFPS)

        # Transforming Images
        (newImages, newTimeElapsed) = reformatImages(newImages, newCorners, newFirstImageXY)

        print("code beyond this point should not work")
        quit
        # appending newly collected items to original arrays
        images = initialImages + newImages
        images = reduceToLastNIndices(images, frametotal)

        faceCorners = np.append(initialCorners, newCorners)
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