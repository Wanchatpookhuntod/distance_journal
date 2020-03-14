import cv2
import numpy as np


if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)

    while True:

        frame = cap.read()[1]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0),(100,100), (0,0,0), -1)
        cv2.line(frame, (0,240), (640,240), (0,0,0),1)

        opicity = 0.4
        cv2.addWeighted(overlay, opicity, frame, 1-opicity, 0, frame)
        

        cv2.imshow("output", frame)
        if cv2.waitKey(1) == 27:
            break


    cv2.destroyAllWindows()
    cap.release()