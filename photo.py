"""photo.py

Snaps a photo from the webcam and crops out only what is 
within the green box. Cropped image is then saved to a 
chosen directory. Use this script to create data for the 
ConvNet. I used this script many times, it was quite 
useful for me.

Spencer Kraisler
""" 

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

saved_dir = "./data/train/digits0/"

# upper left and lower right corner coordinates of region of interest
p1 = (150, 150)
p2 = (400, 430)

i = 0
while(i < 1000):
    _, frame = cap.read()
    box_image = cv2.cvtColor(frame.copy()[p1[0]:2[0], p1[1]:p2[1]], cv2.COLOR_BGR2GRAY)
    cv2.imwrite(saved_dir + "img_%d.jpg" % i, box_image)
    i += 1

    if i % 10 == 0:
    	cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
    	cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()