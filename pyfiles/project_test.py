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
from scipy.ndimage.measurements import label
from project_utils import *

y_start_stop = [400, 656] # Min and max in y to search in slide_window()

image = mpimg.imread('../test_images/test1.jpg')

print(image.shape)
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

bbox_list = []
for scale in [1,1.5,2]:
    bblist = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bbox_list = bbox_list + bblist

heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat,bbox_list)
heat = apply_threshold(heat,1)         
heatmap = np.clip(heat, 0, 255)
labels = label(heatmap)
draw_img = draw_labeled_bboxes(draw_image, labels)


fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
