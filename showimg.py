#!/usr/bin/env python3.6

# importing cv2
import cv2
import sys

if len(sys.argv) != 2:
    print('Usage: %s <file>'%(sys.argv[0]))
    sys.exit(1)

# path
path = sys.argv[1]

# Using cv2.imread() method
# Using 0 to read image in grayscale mode
img = cv2.imread(path, 0)
if not img.any():
    print("Could not open or find the image")
    sys.exit(-1)
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Display window", img )
cv2.waitKey(0)
sys.exit(0)
