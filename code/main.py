import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from function_lib import *

# parameter for training
folder_cars = '../vehicles/'
folder_non_cars = '../non-vehicles/'
svc_file = 'svc.pickle'
features_file = 'features.pickle'
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 24
pix_per_cell = 8
cell_per_block = 1
hog_channel = 'ALL'
spatial_size = (40, 40)
hist_bins = 40

# train classifier and save svc in pickle file
train_clf = False
if train_clf:
    train_classifier(folder_cars, folder_non_cars, svc_file, features_file,
                     force_gen_feature=False, force_gen_svc=False,
                     colorspace=color_space, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                     hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

# plot HOG
if False:
    plot_hog_feature(folder_cars, folder_non_cars, orient, pix_per_cell, cell_per_block)

# test images
img_dir = '../test_images/'
output_dir = '../output_images/'

# get image files
all_files = os.listdir(img_dir)
img_files = [file_names for file_names in all_files if not file_names[0] == '.']
#out_file_name = output_dir + 'heatmaps/heatmap_'
#img_files = ['test1.jpg']
out_file_name = output_dir + 'final/processed_'

for img_file in img_files:
    img = mpimg.imread(img_dir+img_file)

    xstart = 0
    xstop = 1280
    ystarts = [400, 400, 400]
    ystops = [656, 656, 528]

    # plot search area
    if False:
        img = draw_boxes(img, [((xstart, ystarts[0]), (xstop, ystops[0]))], color=(0, 0, 255), thick=6)
        img = draw_boxes(img, [((xstart, ystarts[1]), (xstop, ystops[1]))], color=(0, 255, 255), thick=4)
        img = draw_boxes(img, [((xstart, ystarts[2]), (xstop, ystops[2]))], color=(255, 255, 0), thick=2)

        cv2.imwrite(output_dir + 'search_area.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw_img, heatmap = process_image(img)

    #cv2.imwrite(out_file_name + img_file, heatmap)
    cv2.imwrite(out_file_name+img_file, cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    #plt.imshow(draw_img)
    #plt.show()