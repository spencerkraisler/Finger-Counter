"""
run_ssd_cnn.py

This script uses a pre-trained single shot detector
to draw a bounding box around a hand. Then that box
is cropped out and fed to a conv net to determine
how many digits are being held out. Works best with
the cnn.h5 weights.

NOTE: This script is quite laggy for the first five 
seconds while everything's booting. After that it
works pretty well.

Press Q to quit. Works best against a white wall
and in a well lit area.

Spencer Kraisler 2019
""" 

import cv2
import os
import tensorflow as tf 
import numpy as np
import multiprocessing
from multiprocessing import Queue, Pool
from utils import detector_utils as detector_utils
from utils import pose_classification_utils as classifier


score_thresh = 0.2
im_width, im_height = 1280, 720
num_hands_detect = 1

def worker(input_q, output_q, cropped_output_q, inferences_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)

    print(">> loading keras model for worker")
    try:
        model, classification_graph, session = classifier.load_KerasGraph("./cnn/cnn.h5")
    except Exception as e:
        print(e)

    while True:
        #print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # draw bounding boxes
            detector_utils.draw_box_on_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            
            # classify hand pose
            if res is not None:
                res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
                res = cv2.resize(res, (28, 28), interpolation=cv2.INTER_AREA)
                class_res = classifier.classify(model, classification_graph, session, res)
                inferences_q.put(class_res)       
            
            # add frame annotated with bounding box to queue
            cropped_output_q.put(res)
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()

input_q = Queue(maxsize=5)
output_q = Queue(maxsize=5)
cropped_output_q = Queue(maxsize=5)
inferences_q = Queue(maxsize=5)

cap_params = {}
frame_processed = 0
cap_params['im_width'], cap_params['im_height'] = (1280, 720)
cap_params['score_thresh'] = score_thresh
cap_params['num_hands_detect'] = 1

pool = Pool(5, worker,(input_q, output_q, cropped_output_q, inferences_q, cap_params, frame_processed))

detection_graph, sess = detector_utils.load_inference_graph()

def load_graph(frozen_graph_filename):
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name='prefix')
	return graph

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            output_frame = output_q.get()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            cropped_output = cropped_output_q.get()

            inferences = None
            try:
                inferences = inferences_q.get_nowait()
            except Exception as e:
                pass  

            if inferences is not None:
                pred = np.argmax(inferences) 
                text = "fingers: " + str(pred) + " - " + str((inferences[pred] * 100).astype('uint8')) + "%"
                cv2.putText(output_frame, text, org=(150, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(0,200,0))
            else: 
                cv2.putText(output_frame, "No hand detected", org=(150, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(0,0,200))
            if cropped_output is None:
                cropped_output = np.zeros((100, 100, 1))

            cropped_output = cv2.resize(cropped_output, (100, 100))
            cropped_output = np.resize(cropped_output, (100, 100, 1))
            output_frame[0:cropped_output.shape[0], 0:cropped_output.shape[1]] = cropped_output

            cv2.imshow('frame', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()








