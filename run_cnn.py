"""
run_cnn.py

This script uses a convolution neural network to 
detect how many fingers you're holding out. I built
it through lots and lots of trial and error. 
Works best with cnn2.h5. Put your hand in the green 
square for detection. 

Press Q to quit. Works best against a white wall
and in a well lit area.

Spencer Kraisler 2019
"""

import cv2
cap = cv2.VideoCapture(0)
import os
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Softmax
from keras.layers import Activation, Dropout, Flatten, Dense, ReLU
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.optimizers import Adam

input_shape = (28, 28, 1)
num_classes = 6

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
model.add(ReLU())
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(ReLU())
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(ReLU())
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Softmax())
model.summary()
model.load_weights("./cnn/cnn2.h5")

p1 = (150, 150)
p2 = (400, 430)

while(True):
	_, frame = cap.read()
	box_image = cv2.cvtColor(frame.copy()[p1[0]:p2[0], p1[1]:p2[1]], cv2.COLOR_BGR2GRAY)
	box_image_exp = cv2.resize(box_image, (28, 28))
	box_image_exp = np.expand_dims(box_image_exp, 2)
	box_image_exp = np.expand_dims(box_image_exp, 0)
	inferences = model.predict(box_image_exp)[0]
	prediction = np.argmax(inferences)

	text = "fingers: " + str(prediction) + " - " + str((inferences[prediction] * 100).astype('uint8')) + "%"
	cv2.putText(frame, text, org=(150, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=5, color=(0,200,0))
	box_image = cv2.resize(box_image, (100, 100))
	box_image = np.resize(box_image, (100, 100, 1))

	frame[0:box_image.shape[0], 0:box_image.shape[1]] = box_image
	cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()