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
    camera.brightness = 100
    camera.framerate = framerate

    time.sleep(0.1)

    rawCapture = picamera.array.PiRGBArray(camera)
    camera.capture(rawCapture, format="bgr")
    
    # Start timer
    t_start = time.time()
        
    while 1: 
        # Gets and stores images
        for i in range(i, i+10):
            # capture not working right, seems like it only captures one frame several times
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
                # Get RGB intensities
                r[j] = numpy.sum(images[j][:][:][3])
                g[j] = numpy.sum(images[j][:][:][2])
                b[j] = numpy.sum(images[j][:][:][1])

                # Detrend and normalize RGB intensities
                # detrend isn't used correctly I think
                #r_norm = scipy.signal.normalize(scipy.signal.detrend(r), [1])
                #g_norm = scipy.signal.normalize(scipy.signal.detrend(g), [1])
                #b_norm = scipy.signal.normalize(scipy.signal.detrend(b), [1])

        if framenum >= 240: #frametotal:
            camera.close()
            t_finish = time.time()
            t_total = t_finish-t_start
            print("time to run : %.3f" % (t_total), "(s)")
            break


    ### TESTING ###  ### TESTING ### 

    ### TESTING ###  ### TESTING ### 

def GetImage(images, i, framenum, rawCapture):
    print("yes")
    for i in range(i, i+10):
        images.append(rawCapture.array)
        time.sleep(1/60)
        framenum += 1
    




if __name__ == "__main__":
    main()

