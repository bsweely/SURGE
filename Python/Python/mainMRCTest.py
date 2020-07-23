from io import BytesIO
import time
import numpy
import picamera
import picamera.array
import io
import scipy
import scipy.signal
import imageio

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
    camera.resolution = (640, 480)
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
        for image in range(len(images)):
            images[image] = imageio.imread(images[image]) # makes numpy RGb array by default

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