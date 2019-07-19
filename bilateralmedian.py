
# USAGE
# python detect_bright_spots.py --image images/lights_01.png

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments


# load the image, convert it to grayscale, and blur it
image = cv2.imread("E:\\internship\\B. Disease Grading\\B. Disease Grading\\1. Original Images\\a. Training Set\\IDRiD_048.jpg")
image=cv2.resize(image,(800,615))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray,9,75,75)
median=cv2.medianBlur(blur,5)


cv2.imshow("median",median)

#cv2.imshow()

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(median, 155, 255, cv2.THRESH_BINARY)[1]
# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue

	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels >300:
		mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]
# loop over the contours
for (i, c) in enumerate(cnts):
     ellipse = cv2.fitEllipse(c)
     (x, y, w, h) = cv2.boundingRect(c)
     cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
     cv2.ellipse(image,ellipse,(0,255,0),2)
          
     print(h)
     print(w)
     print(x)
     print(y)
     break
	# draw the bright spot on the image
	
cv2.imshow("Image", image)
cv2.waitKey(0)