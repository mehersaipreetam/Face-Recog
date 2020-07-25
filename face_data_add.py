import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


frames = []
output = []

while True:
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x,y,w, h = face
            clip = frame[y:y + h,x:x + w]

            fixed_clip = cv2.resize(clip, (200,200))

            gray = cv2.cvtColor(fixed_clip,
            cv2.COLOR_BGR2GRAY)

            cv2.imshow('My face',gray)

        #cv2.imshow('My screen', frame)

    key = cv2.waitKey(1)
    name = input("Enter your name: ")
    if key == ord("q"):  # Unicode conversion of 'q'
        print('Coming out')
        break

    if key == ord("c"): # capture
        #cv2.imwrite(name + '.jpg',frame)
        frames.append(gray.flatten())
        output.append([name])

X = np.array(frames)
y = np.array(output)

# also we have to save the data, we cant run everything at a time
# name with falttened image
data = np.hstack([y,X])
file_name = 'face_data.npy'

if os.path.exists(file_name):
    old = np.load(file_name)
    data = np.vstack([old, data])

np.save(file_name, data)

cap.release()
cv2.destroyAllWindows()
