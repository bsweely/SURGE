import numpy
import picamera
import picamera.array
import time
import io
import scipy
import scipy.signal



def main():
    # Variables:
    framenum   = 0
    framerate  = 120
    frametotal = 240

    images = []
    r = numpy.zeros(60)
    g = numpy.zeros(60)
    b = numpy.zeros(60)

    # Iterators
    i = 0
    j = 0
    k = 0

    # Connect to camera
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    # camera.start_preview() # inserted. Perhaps not needed.
    camera.brightness = 100
    camera.framerate = framerate

    # time.sleep(0.1)
    time.sleep(2) # inserted from online tutorial

    # rawCapture = picamera.array.PiRGBArray(camera)
    # camera.capture(rawCapture, format="bgr")

    # Start timer
    t_start = time.time()

    while 1:
        # Gets and stores images
        for i in range(i, i+10):
            # capture not working right, seems like it only captures one frame several times, might be with getting RGB intensities
            # I didn't see it either. The following line of code captures a new
            # picture, but with it outside of the loop, it was just appending
            # the same picture over and over again without taking a new one
            rawCapture = picamera.array.PiRGBArray(camera)
            camera.capture(rawCapture, format="bgr")
            images.append(rawCapture.array)
            time.sleep(1/60)
            framenum += 1

        i+=1

        if framenum == 60:
            t_capture = time.time()
            print("capture rate: %.2f" % (framenum/(t_capture-t_start)), "(Hz)")

        # Limits stored images to 60
        if i == 60:
            i = 0
            j = 0
            for j in range(j, j+60):
                # Find ROI and crop array

                # Get RGB intensities
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
    for i in range(i, i+10):
        images.append(rawCapture.array)
        time.sleep(1/60)
        framenum += 1

if __name__ == "__main__":
    main()