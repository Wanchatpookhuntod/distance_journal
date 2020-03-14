import cv2
import dlib
import numpy as np
from find_angle_distance import Find_angle_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Face_detect:
    model_hog = dlib.get_frontal_face_detector()
    model_haar = cv2.CascadeClassifier("res/model/haarcascade_frontalface_default.xml")
    f_txt = "res/model/opencv_face_detector.pbtxt"
    f_pb = "res/model/opencv_face_detector_uint8.pb"
    model_dnn = cv2.dnn.readNetFromTensorflow(f_pb, f_txt)
    predict_landmark = dlib.shape_predictor("res/model/shape_predictor_5_face_landmarks.dat")

    find_distance = Find_angle_distance()

    def get_frame(self, frame):
        self.frame = frame

    def haar(self, model):
        dets = model.detectMultiScale(self.frame, scaleFactor=1.3, minNeighbors=5)

        self.face_list = []
        for x, y, w, h in dets:
            left, top, right, bottom = x, y, x + w, y + h
            self.face_list.append([left, top, right, bottom])
        return self.face_list

    def hog_dlib(self, model):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        dets = model(gray, 0)

        self.face_list = []
        for det in dets:
            self.face_list.append([det.left(), det.top(), det.right(), det.bottom()])
        return self.face_list

    def dnn(self, model):
        rows = self.frame.shape[0]
        cols = self.frame.shape[1]
        model.setInput(cv2.dnn.blobFromImage(self.frame, size=(300, 300), swapRB=True, crop=False))
        face_Out = model.forward()

        self.face_list = []
        for detection in face_Out[0, 0, :, :]:
            score = float(detection[2])

            if score < 0.6:
                continue
            box = detection[3:7] * np.array([cols, rows, cols, rows])
            (left, top, right, bottom) = box.astype("int")

            origin_w_h = bottom - top
            top = int(top + (origin_w_h) * 0.15)
            bottom = int(bottom - (origin_w_h) * 0.05)
            margin = ((bottom - top) - (right - left)) // 2
            left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1
            right = right + margin
            self.face_list.append([left, top, right, bottom])
        return self.face_list

    def method_face(self, select_method):
        if select_method == 0:
            self.face = self.haar(self.model_haar)

        elif select_method == 1:
            self.face = self.hog_dlib(self.model_hog)
        else:
            self.face = self.dnn(self.model_dnn)
        return self.face

    def face_landmark(self, face):
        if face:
            # l,t,r,b = face[-1][0:len(face)]
            for l,t,r,b in face:

                shape = self.predict_landmark(self.frame, dlib.rectangle(int(l),int(t),int(r),int(b)))

                extand_eye = lambda  near, far: (far - near) // 2 + near

                r_e_x = extand_eye(shape.part(2).x, shape.part(3).x)
                r_e_y = extand_eye(shape.part(2).y, shape.part(3).y) \
                    if shape.part(3).y > shape.part(2).y else extand_eye(shape.part(3).y, shape.part(2).y)

                l_e_x = extand_eye(shape.part(1).x, shape.part(0).x)
                l_e_y = extand_eye(shape.part(1).y, shape.part(0).y) \
                    if shape.part(0).y > shape.part(1).y else extand_eye(shape.part(0).y, shape.part(1).y)

                c_x = extand_eye(r_e_x, l_e_x)
                c_y = extand_eye(r_e_y, l_e_y)

                points = {"view_center":[c_x, c_y], "eye_right":[r_e_x, r_e_y], "eye_left":[l_e_x, l_e_y]}
                return points

    def draw_face(self, face, t):
        # for points_faces in range(len(face)):
        #     x,y,w,h = face[points_faces][0:4]
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_inpurt = f'{t:.2f} cm'
        s_text = 0.8

        box = cv2.getTextSize(t_inpurt, font, s_text, 1)
        w_t = box[0][0]
        h_t = box[0][1]

        for x,y,w,h in face:
            cv2.rectangle(self.frame, (x,y), (w,h), (0,0,0), 1)
            cv2.rectangle(self.frame,(x,y-8), (x+w_t, ((y-12)-h_t)), (255,255,255), -1 )
            cv2.putText(self.frame,t_inpurt,(x,y-10),font, s_text, (0,0,255), 1)

    def draw_landmark(self, eye):
        cv2.circle(self.frame,(eye[0],eye[1]), 2, (0,0,0), -1)

    def distance(self, view_center, eye_right):
        self.find_distance.get_eye(view_center)
        setcam = self.find_distance.set_camera()
        status_v = self.find_distance.change_point_start_vertical()
        angle_v = self.find_distance.estimate_angle_vertical()
        distance = self.find_distance.estimate_distance(eye_right[0])
        return {"distance": distance, "angle": [angle_v, status_v[0]]}

time_processing = lambda x: (cv2.getTickCount() - x) / cv2.getTickFrequency() * 1000

def choise_name(choose, dis):
    name = ""
    if choose == 0:
        name = "HAAR"
    elif choose == 1:
        name = "HOG"
    else:
        name = "DNN"
    return f"{name}_{dis}"


if __name__ == "__main__":
    import argparse
    _c = 2


    
    _d = 80
    stop = 1000


    parser = argparse.ArgumentParser()
    parser.add_argument('--m', help='choose face roi, 0(haar) 1(hog) 2(dnn)', type=int, default= _c)
    parser.add_argument('--d', help='distance detect)', type=int, default= _d)
    args = parser.parse_args()

    choose = args.m
    d = args.d
    name = choise_name(choose, d)

    cap = cv2.VideoCapture(0)
    face_detect = Face_detect()
    frame_num = 0
    speed_list = []

    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    dist_list = []


    while True:
        frame = cap.read()[1]

        start = cv2.getTickCount()

        face_detect.get_frame(frame)
        face = face_detect.method_face(choose)
        eyes = face_detect.face_landmark(face)

        if eyes:
            frame_num += 1

            face_detect.draw_landmark(eyes["view_center"])
            dist = face_detect.distance(eyes["view_center"], eyes["eye_right"])['distance']
            face_detect.draw_face(face, dist)

            dist_list.append(dist)

            speed = time_processing(start)
            speed_list.append(speed)

        cv2.putText(frame, f"{name} | Speed: {speed:.2f} ",(20,30), font, 0.6, white, 1)
        cv2.putText(frame, f"Frame: {frame_num} ",(20,60), font, 0.6, white, 1)

        cv2.imshow("out", frame)
        if cv2.waitKey(1) == 27 or frame_num == stop:
            break
    
    cv2.destroyAllWindows()
    cap.release()

    # np.savetxt(f"csv/new_test/{name}.csv", dist_list)

    # y_true = np.zeros(stop) + d

    # print("\n")
    # print(f"MSE: {mean_squared_error(y_true, dist_list, squared= False)}")
    # print(f"RMSE: {mean_squared_error(y_true, dist_list)}")
    print(f"SPEED:{np.array(speed_list).mean():.2f}")
