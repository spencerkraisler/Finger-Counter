import numpy as np
import cv2
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Softmax
from keras.layers import Activation, Dropout, Flatten, Dense, ReLU
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam

MODE = "CONFUSION_MATRIX"

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
model.load_weights("cnn.h5")

if MODE == "VAL_SET":
	test_data_dir = "../data/val"
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
			test_data_dir,
			color_mode='grayscale',
			shuffle=False,
			target_size=(input_shape[0], input_shape[1]),
			class_mode='categorical')
	score = model.evaluate_generator(test_generator, steps=50, verbose=1)
	print(score)

elif MODE == "CONFUSION_MATRIX":
	test_data_dir = "../data/val"
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
			test_data_dir,
			color_mode='grayscale',
			shuffle=False,
			target_size=(input_shape[0], input_shape[1]),
			class_mode='categorical')
	Y_pred = model.predict_generator(test_generator, steps=len(test_generator))
	y_pred = np.argmax(Y_pred, axis=1)
	print(confusion_matrix(test_generator.classes, y_pred))


