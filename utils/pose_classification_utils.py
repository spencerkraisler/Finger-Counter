import cv2
import numpy as np
from tensorflow import Graph, Session
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Softmax
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, ReLU
from tensorflow.keras.optimizers import Adam

def load_KerasGraph(path): 
    print("> ====== loading Keras model for classification")
    thread_graph = Graph()
    with thread_graph.as_default():
        thread_session = Session()
        with thread_session.as_default():
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
            model.load_weights(path)

            graph = tf.get_default_graph()
    print(">  ====== Keras model loaded")
    return model, graph, thread_session

def classify(model, graph, sess, im):
    res = cv2.flip(im, 1)

    # Reshape
    # res = cv2.resize(im, (28,28), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float64")
    res = res / 255
    res = np.reshape(res, (1, 28, 28, 1))

    with graph.as_default():
        with sess.as_default():
            prediction= model.predict(res)

    return prediction[0] 
