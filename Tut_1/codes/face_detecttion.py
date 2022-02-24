import cv2 as cv

## step 1: load image 
image = cv.imread('Tut_1\images\people2.jpg')


## scale image 
scale_factor = 1
if scale_factor == 1:
    image_scaled = image
else:
    height, width = image.shape
    height *= scale_factor ; width *= scale_factor
    image_scaled = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

## grey scaling 
image_grey = cv.cvtColor(image_scaled, cv.COLOR_BGR2GRAY)

## Define detector
face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
face_detector.detectMultiScale(image_grey)

cv.imshow('original', image)
cv.imshow('Scaled', image_scaled)
cv.imshow('Gray', image_grey)
cv.waitKey(0)





