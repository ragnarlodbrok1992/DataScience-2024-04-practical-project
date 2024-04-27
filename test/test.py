import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread('ML-datasets/images/lena.png')
cv2.imshow('image', img)

# input()
cv2.waitKey(0)

# What to import for it to work
