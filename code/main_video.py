import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from moviepy.editor import VideoFileClip
from function_lib import *

folder = "../"
output_folder = "../output_images/"
#input_video_file = "test_video.mp4"
input_video_file = "project_video.mp4"

output_video_file = output_folder + "processed_" + input_video_file

clip = VideoFileClip(folder + input_video_file)

# process the video
output_clip = clip.fl_image(process_image)

output_clip.write_videofile(output_video_file, audio=False)
