#test that intel realsense camera works
import cv2

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    #q to quit
    if cv2.waitKey(1) & ord('q'):
        break

#release the video
vid.release()
cv2.destroyAllwindows()
