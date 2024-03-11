# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np

input_dir = 'dataset/input'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
def segmentImage(inputImg):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    # RGB to HSV
    hsv = cv2.cvtColor(inputImg, cv2.COLOR_RGB2HSV)
    # image denoise
    denoise = cv2.fastNlMeansDenoisingColored(hsv,None,10,10,7,21)
    # image bluring
    blur = cv2.medianBlur(denoise,23)
    # Using H channel to segmentation process
    H = blur[:,:,0]
    hsv_img = H

    kernel = np.ones((5,5),np.uint8)

    # Find white blood cells
    # define range of white blood cell color in H channel
    lower_white = np.array([155])
    upper_white = np.array([255]) 
    # Find the range of wbc 
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white) 
    # Morphology Open 
    opening = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations = 3)
    # Dilate
    dilate = cv2.dilate(opening,kernel,iterations = 2)
    seg_wbc=dilate # segment done

    # crop area white blood cells
    crop_wbc = cv2.subtract(hsv_img,seg_wbc)

    # Find red blood cells
    # define range of red blood cell color in H channel
    lower_red = np.array([85])
    upper_red = np.array([220])
    # Find the range of rbc 
    mask_red = cv2.inRange(crop_wbc, lower_red, upper_red) 
    # Morphology Open 
    opening = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations = 2)
    # Dilate
    dilate = cv2.dilate(opening,kernel,iterations = 1)
    # Erosion
    erode = cv2.erode(dilate,kernel,iterations = 2)
    seg_rbc=erode # segment done

    seg_wbc = seg_wbc/255
    seg_rbc = seg_rbc/255*2

    # merge wbc & rbc
    merge_mask = cv2.bitwise_or(seg_wbc, seg_rbc)
    outputImg = merge_mask

    # END OF YOUR CODE
    #########################################################################
    return outputImg

