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

# test images
img_dir = '../test_images/'
output_dir = '../output_images/'

# get image files
all_files = os.listdir(img_dir)
img_files = [file_names for file_names in all_files if not file_names[0] == '.']
#img_files = ['test1.jpg']
out_file_name = output_dir + 'raw/marked_'

# get classifier
with open(svc_file, "rb") as f:
    svc_data = pickle.load(f)
svc = svc_data["svc"]
X_scaler = svc_data["scaler"]

ystarts = [400, 400, 400]
ystops = [656, 656, 528]
scales = [1.5, 1, 0.75]

for img_file in img_files:
    img = mpimg.imread(img_dir+img_file)
    # find car on image
    vehicles_boxes = []
    for ystart, ystop, scale in zip(ystarts, ystops, scales):
        v_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if len(v_boxes) > 0:
            for box in v_boxes:
                vehicles_boxes.append(box)
    draw_img = draw_boxes(img, vehicles_boxes)
    #draw_img = generate_heatmap(img, vehicles_boxes)

    cv2.imwrite(out_file_name+img_file, cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    #plt.imshow(draw_img)
    #plt.show()