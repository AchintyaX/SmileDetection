# importing the libraries 
import cv2

#loading the cascades 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#defining the function
def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle( frame, (x, y), (x+w, y+h ), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x:x+w]
        roi_color = frame[y: y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22, minSize=(25, 25), )
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame 

# doing face recognition and smile detection using the webcam 
video_capture = cv2.VideoCapture(0) # 0 for the internal webcam
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break 
    
video_capture.release()
cv2.destroyAllWindows()