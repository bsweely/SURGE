import numpy
import picamera
import picamera.array

def main():
    # Variables:
    framenum   = 0
    framerate  = 60
    frametotal = 240
    images     = [] # Create array of images {image,x,y,rgb}

    
    # Iterators
    i = 0
    j = 0
    k = 0


    # Connect to camera
    camera = picamera.PiCamera()
    # camera.resolution = (640, 480)
    camera.resolution = (64, 64)
    camera.brightness = 100
    camera.framerate = framerate
    
    
    
    while 1: 
        for i in range(i, i+10): # ( ; i < (j+10); i++)
            # img = picamera.array.PiRGBArray(camera)
            # camera.capture(img, 'rgb')
            # images.append(img)
            # img.truncate(0)
            framenum += 1
            # take and store image
            # get rgb signal channels

        # Stores 60 frames
        #if i == 60:
        #    i = 0
        #    j = 0
        
        

        print(framenum)
        if framenum == frametotal:
            break



    ### TESTING ###  ### TESTING ### 

    # image = camera.capture(output, 'rgb')
    img = numpy.zeros((64,64,3))
    img = picamera.array.PiRGBArray(camera)
    camera.capture(img, 'rgb')
    numpy.append(images,img, 0)
    # igg2.append(img)
    img.truncate(0)
    camera.capture(img, 'rgb')
    images.append(img)
    # camera.capture_sequence([images for i in range(10)])

    print(igg2)
    print(images)

    if images[0] == images[1]:
        print("Equal")
    else:
        print("Not Equal")

    # framenum, image x, image y, rgb
    print('Captured Image[%d] %dx%dx%d ' % (framenum, img.array.shape[1], img.array.shape[0], img.array.shape[2]))

    ### TESTING ###  ### TESTING ### 

    

if __name__ == "__main__":
    main()
