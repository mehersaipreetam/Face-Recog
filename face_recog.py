import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)

data = np.load('face_data.npy')

X = data[:,1:].astype(int)
y = data[:,0]

model = KNeighborsClassifier()
model.fit(X,y)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# name = input("Enter your name: ")
# frames = []
# output = []

while True:
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x,y,w,h = face
            clip = frame[y:y + h,x:x + w]

            fixed_clip = cv2.resize(clip, (200,200))

            gray = cv2.cvtColor(fixed_clip,
            cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()])
            
            frame_w_rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(100,0,0),2)
            
            cv2.putText(frame_w_rect, str(out[0]), (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    key = cv2.waitKey(1)
        
    cv2.imshow('My screen', frame)     
            
    if key == ord("q"):  # Unicode conversion of 'q'
            print('Coming out')
            break

cap.release()
cv2.destroyAllWindows()
