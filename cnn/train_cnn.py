import cv2
import numpy as np 
from keras.models import Sequential, Model
from keras import metrics
from keras.layers import Conv2D, MaxPooling2D, Softmax
from keras.layers import Activation, Dropout, Flatten, Dense, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


optimizer = Adam(lr=.005)
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
model.load_weights("cnn2.h5")
model.summary()
model.compile(
		loss='categorical_crossentropy',
		optimizer=optimizer,
		metrics=[metrics.categorical_accuracy])

batch_size = 128
train_data_dir = "../data/train"
val_data_dir = "../data/val"

train_datagen = ImageDataGenerator(
		rescale=1./255,
		brightness_range=(0.0, 1.0),
		rotation_range=90,
		shear_range=0.2,
		horizontal_flip=True,
		width_shift_range=.1,
		height_shift_range=.05,
		zoom_range=.2,
		fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		color_mode="grayscale",
		shuffle=True,
		target_size=(input_shape[0], input_shape[1]),
		batch_size=batch_size,
		class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
		val_data_dir,
		color_mode="grayscale",
		shuffle=True,
		target_size=(input_shape[0], input_shape[1]),
		batch_size=batch_size,
		class_mode='categorical')


model.fit_generator(
		train_generator,
		shuffle=True,
		class_weight='balanced',
		steps_per_epoch=2000 // batch_size,
		epochs=10,
		validation_data=val_generator,
		validation_steps=800 // batch_size)

model.save_weights('./cnn2.h5')

