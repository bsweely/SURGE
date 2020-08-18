# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:28:07 2020

@author: jrsho
"""


import numpy as np
import time
import matplotlib.pyplot as plt
import cv2 as cv

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


from picamera import PiCamera

'''
camera = PiCamera()

camera.start_preview()
time.sleep(5)
camera.capture('/home/pi/Desktop/imageFromTestCode.jpg')
camera.stop_preview()
'''

# Testing the camera.capture_sequence() method
# import time
# import picamera

with PiCamera() as camera:
    camera.start_preview()
    time.sleep(2)
    images = ['image1', 'image2', 'image3']
    '''
    camera.capture_sequence([
        'image1.jpg',
        'image2.jpg',
        'image3.jpg',
        ])
    '''
    frames = camera.capture_sequence(images)
    camera.stop_preview()
print(type(images))
print("Type of the objects in the images list: ", type(images[0]))
print("The type of object that capture_sequence captures", type(frames))

for image in range(len(images)):
    img = cv.imread(images[image])
    print("This is the type for the image once it has been read with cv2: ", type(img))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    corners = np.float32(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    fig = plt.figure()
    plt.imshow(img),plt.show()


