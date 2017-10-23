#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:16:49 2017

@author: subbu
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

clip = VideoFileClip('../test_video.mp4')

integration_length = 6
heat_history = [None for ii in range(integration_length)]
images = []
for image in clip.iter_frames():
    images.append(image)
ind = 0  
#imagefiles = ['../test_images/test1.jpg',
#          '../test_images/test2.jpg',
#          '../test_images/test3.jpg',
#          '../test_images/test4.jpg',
#          '../test_images/test5.jpg',
#          '../test_images/test6.jpg']
#images = []
#for imagefile in imagefiles:
#    images.append(mpimg.imread(imagefile))
for image in images:
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    bbox_list = find_cars(image, 400, 650, 950, 1280, 2.0)
    bbox_list += find_cars(image, 400, 500, 950, 1280, 1.5)
    bbox_list += find_cars(image, 400, 650, 0, 330, 2.0)
    bbox_list += find_cars(image, 400, 500, 0, 330, 1.5)
    bbox_list += find_cars(image, 400, 460, 330, 950, 0.75,3)

    current_heat = np.zeros_like(image[:,:,0]).astype(np.float)
    current_heat = add_heat(current_heat,bbox_list)
    
    if ind == 0:
        heat = np.copy(current_heat)
    elif ind < integration_length:
        heat += current_heat
    else:
        heat -= heat_history[ind % integration_length]
        heat += current_heat
    heat_history[ind % integration_length] = np.copy(current_heat)
        
    heat = apply_threshold(heat,2)         
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_image1 = draw_boxes(draw_image,bbox_list)
    draw_image = draw_labeled_bboxes(draw_image, labels)
    cv2.imshow('image',draw_image1)
    plt.imshow(heatmap,cmap = 'hot')
    cv2.waitKey(1)
    ind += 1

cv2.destroyAllWindows()
