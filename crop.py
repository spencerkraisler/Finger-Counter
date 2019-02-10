"""
crop.py

This script sifts through an entire dataset of images and 
uses the hand detector to crop out the region of interest,
ideally a hand. The cropped image is saved to a desired
directroy. If hand detector detects no hand, it'll skip onto
the next image.

Spencer Kraisler
"""

import cv2
import os
import tensorflow as tf 
import numpy as np
from utils import detector_utils as detector_utils

detection_graph, sess = detector_utils.load_inference_graph()

score_thresh = 0.27
im_width, im_height = 1280, 720
num_hands_detect = 1

# returns list of every directory and file name within a chosen 
# directory.
def get_names(dir_path):
	names = os.listdir(dir_path)
	for name in names:
		if name == ".DS_Store": names.remove(name)
	return names

# loads tf graph of hand detector
def load_graph(frozen_graph_filename):
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name='prefix')
	return graph

# uses the hand detector to crop out and return the region of interest
# from a selected photo. 
def get_box_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, border):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            if (left - border >= 0 and top - border >= 0):
            	p1 = (int(left) - border, int(top) - border)
            else:
            	p1 = (int(left), int(top))

            if (right + border <= im_width and bottom + border <= im_height):
            	p2 = (int(right) + border, int(bottom) + border)
            else:
            	p2 = (int(right), int(bottom))
            return image_np[p1[1]:p2[1], p1[0]:p2[0]].copy()

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:		
		
		# use this if you want it to sift through a directory of
		# classes of images and crop the region of interests
		# e.g.
			# dir 
				# class1
					# img_1.jpg
					# img_2.jpg
					# ...
				# class 2
					# ...
				# ...

		root_dir = "./data/train"
		classes = get_names(root_dir)
		for class_name in classes:
			image_names = get_names(root_dir + "/" + class_name)
			for image_name in image_names:
				image = cv2.imread(root_dir + "/" + class_name + "/" + image_name)
				boxes, scores = detector_utils.detect_objects(image, detection_graph, sess)
				box_image = get_box_image(1, score_thresh, scores, boxes, 
										 im_width, im_height, image, 50)
				if box_image is not None: 
					cv2.imwrite("./data/cropped_data/train/" + class_name + "/" + image_name, cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY))
		"""
		# use this if you just want it to sift through one specific directory of images
		
		root_dir = "./data/data4/train"
		
		class_name = "digits0"
		image_names = get_names(root_dir + "/" + class_name)
		for image_name in image_names:
			image = cv2.imread(root_dir + "/" + class_name + "/" + image_name)

			boxes, scores = detector_utils.detect_objects(image, detection_graph, sess)
			box_image = get_box_image(1, score_thresh, scores, boxes, 
										 im_width, im_height, image, 50)
			if box_image is not None: 
				cv2.imwrite("./data/cropped_data4/train/" + class_name + "/" + image_name, cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY))
		"""
