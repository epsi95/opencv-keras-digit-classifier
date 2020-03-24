# The code is written by probhakar sarkar on 24th March, 2020

from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
import matplotlib
import imutils
import time
import cv2

input_device = int(input("Press 0 for laptop camera, 1 for USB camera: "))
# To capture video from webcam.
cap = cv2.VideoCapture(input_device)

# training neural net for detecting handwritten digit
print('Model training started ---------->>>')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape= (28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('accuracy of the trained model: ', test_acc)
print('Model training completed ---------->>>')

shouldWrite = False # variable to keep track whether user draw or not
points = [] # variable to store the drawn points
predicted_value = "_._"

def getBoundingBox(points): # function to get the Top Left corner point and Bottom Right corner point
    bbTL = (min([i[0] for i in points]), min([i[1] for i in points]))
    bbBR = (max([i[0] for i in points]), max([i[1] for i in points]))
    return([bbTL, bbBR])

print('<<---IMPORTANT INFORMATION--->>')
print('press "d" key to start and stop drawing.')
print('press "q" to stop.')
print('use blue colored object for tracking')
print('UI is designed for frame shape of 480, 640, change the camera accordingly, otherwise things may overlap.')
print('<<---END--->>')
while True:
    # grabbing the frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    X = np.shape(frame)[1] # width of frame
    Y = np.shape(frame)[0] # height of frame

    # defining bounding box for the user drawing space
    TLp = (int(X*(2/3)), int(Y*(1/3)))
    BRp = (TLp[0] + 200, TLp[1] + 200)
    # defining bounding box for the button
    TLp2 = (int(X * (2 / 3)), int(Y * (1 / 3)) + 210)
    BRp2 = (TLp2[0] + 200, TLp2[1] + 40)
    # defining bounding box for status update
    TLp3 = (int(X * (2 / 3)), int(Y * (1 / 3)) + 210 + 50)
    BRp3 = (TLp3[0] + 200, TLp3[1] + 40)

    cv2.rectangle(frame, TLp, BRp, (255, 255, 255), 1) # user drawing space
    cv2.rectangle(frame, TLp2, BRp2, (102, 0, 255), 1) # button
    cv2.putText(frame, "Predict",
                (TLp2[0] + 40,  TLp2[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # button text
    cv2.rectangle(frame, TLp3, BRp3, (255, 255, 255), -1) # status update space
    cv2.putText(frame, "value: {}".format(predicted_value),
                (TLp3[0] + 20, TLp3[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2) # status update space text

    # color range for detection
    # find more here: https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
    blueLower = (110, 50, 50)
    blueUpper = (130, 255, 255)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "blue", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the detected area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if shouldWrite:
            points.append(center)
        if len(points) > 1:
            for pp in zip(points, points[1:]):
                cv2.line(frame, pp[0], pp[1], (0, 0, 255), 5)
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if(center[0] > TLp2[0] and center[1] > TLp2[1] and center[0] < BRp2[0] and center[1] < BRp2[1]): # detect button press
                # print("pressed")
                cv2.rectangle(frame, TLp2, BRp2, (255, 255, 255), -1)
                time.sleep(0.1)
                cv2.rectangle(frame, TLp2, BRp2, (102, 0, 255), -1)
                cv2.putText(frame, "Predict",
                            (TLp2[0] + 40, TLp2[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                if(len(points) > 0):
                    bb = getBoundingBox(points)
                    bbTL = bb[0]
                    bbBR = bb[1]
                    digitRaw = np.ones(((bbBR[1] - bbTL[1]) + 20, (bbBR[0] - bbTL[0]) + 20))
                    digitRaw = digitRaw * 255
                    offsetedPoints = [(i - bbTL[0], j - bbTL[1]) for i, j in points]
                    if len(offsetedPoints) > 1:
                        for pp in zip(offsetedPoints, offsetedPoints[1:]):
                            cv2.line(digitRaw, (pp[0][0] + 10, pp[0][1] + 10), (pp[1][0] + 10, pp[1][1] + 10),
                                     (0, 0, 0), 5)

                    digitRaw = cv2.copyMakeBorder(digitRaw, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    invertedDigit = 255 - digitRaw
                    digit = cv2.resize(invertedDigit, (28, 28), interpolation=cv2.INTER_AREA)
                    #plt.imshow(digit, cmap=matplotlib.cm.binary, interpolation="nearest")
                    #plt.show()
                    prediction_value = ((digit.reshape(28 * 28, )).astype('float32')) / 255
                    # print('shape', np.shape(prediction_value))
                    prediction = network.predict(np.array([prediction_value]))
                    # print('prediction', prediction)
                    print(np.argmax(prediction))
                    predicted_value = str(np.argmax(prediction))
                    cv2.putText(frame, "Predicted value: {}".format(predicted_value),
                                (TLp3[0] + 20, TLp3[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    points.clear()

    c = cv2.waitKey(1)
    cv2.imshow('frame', frame)
    if 'q' == chr(c & 255):
        break
    elif 'd' == chr(c & 255):
        shouldWrite = not shouldWrite
# Release the VideoCapture object
cap.release()