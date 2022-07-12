import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image = cv2.imread('sudoku.png')
imshow('Original Image',image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow('Gray Image',gray)

blur = cv2.GaussianBlur(image,(3,3),0)
imshow('Blurred Image',blur)

### CONTOURS

image = cv2.imread('sudoku.png')
image2 = image.copy()
og_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Finding Contours
# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

# not_image = cv2.bitwise_not(th2)
# imshow('Bitwise NOT on grayscale image', not_image)

#contours2, hierarchy2 = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image2, contours2, -1, (0,0,255), thickness = 2)
imshow('Contours overlaid on Bitwise NOT image', image2)

print("Number of Contours found = " + str(len(contours)))
print("Number of Contours found = " + str(len(contours2)))

zeros = []

for (i,c) in enumerate(contours2):
  x,y,w,h= cv2.boundingRect(c)  
  cropped_contour= og_image[y:y+h, x:x+w]
  digit_contour = cropped_contour.copy()
  #plt.imshow(cropped_contour)
  gray_contour = cv2.cvtColor(cropped_contour, cv2.COLOR_BGR2GRAY)
  _,th2 = cv2.threshold(gray_contour, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  digits, hierarchy2 = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(cropped_contour, digits, 0, (0,255,0), thickness = 1)
  
  #imshow('Contours overlaid on original image', cropped_contour)
  #imshow('Crop', cropped_contour)
  #plt.imshow(cropped_contour)
  #print(cropped_contour.shape)
  if len(digits) > 1:
    image_name= "block_" + str(81-i) + ".jpg"
    cv2.imwrite(os.path.join('images/',image_name), digit_contour)
  else:
    zeros.append(81-i) 
#print(zeros)

def return_zeros():
  return zeros