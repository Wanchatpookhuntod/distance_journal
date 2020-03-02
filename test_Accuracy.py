import cv2
import dlib
import numpy as np
import argparse
from find_angle_distance import Find_angle_distance


parser = argparse.ArgumentParser()
parser.add_argument('--face_roi', help='choose face roi, 0(hog) 1(haar) 2(dnn)', type=int, default=2)
args = parser.parse_args()

predict_landmark = dlib.shape_predictor("res/model/shape_predictor_5_face_landmarks.dat")
time_processing = lambda x: (cv2.getTickCount() - x) / cv2.getTickFrequency() * 1000

def face_landmark(img, left, top, right, bottom):
    shape = predict_landmark(img, dlib.rectangle(int(left), int(top), int(right), int(bottom)))
    points = []
    for i in range(shape.num_parts):
        point = shape.part(i).x, shape.part(i).y
        points.append(point)
    return  points

def hog_dlib(img,model):
    dets = model(img, 1)

    face = []
    for det in dets:
        face.append([det.left(), det.top(), det.right(), det.bottom()])
    return face

def haar(img,model):
    dets = model.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    face = []
    for x, y, w, h in dets:
        left, top, right, bottom = x, y, x + w, y + h
        face.append([left, top, right, bottom])
    return face

def dnn (img, model):
    rows = img.shape[0]
    cols = img.shape[1]

    model.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    face_Out = model.forward()

    face = []
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
        face.append([left, top, right, bottom])
    return face


if __name__ == '__main__':
    find_distance = Find_angle_distance()
    select_method = args.face_roi
    model = ""

    def distance(view_center, eye_right):
        find_distance.get_eye(view_center)
        setcam = find_distance.set_camera()
        status_v = find_distance.change_point_start_vertical()
        angle_v = find_distance.estimate_angle_vertical()
        dis = find_distance.estimate_distance(eye_right[0])
        return {"distance": dis, "angle": [angle_v, status_v[0]]}


    if select_method == 0:
        model = dlib.get_frontal_face_detector()
    elif select_method == 1:
        model = cv2.CascadeClassifier("res/model/haarcascade_frontalface_default.xml")
    elif select_method == 2:
        f_txt = "res/model/opencv_face_detector.pbtxt"
        f_pb = "res/model/opencv_face_detector_uint8.pb"
        model = cv2.dnn.readNetFromTensorflow(f_pb, f_txt)
    else:
        print("please select 0(hog) 1(haar) 2(dnn)")
        exit()

    cap = cv2.VideoCapture(0)

    num_frame = 0
    list_time = []
    text_method = ""
    color = ""

    font = cv2.FONT_HERSHEY_COMPLEX
    color_value = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "white": (255,255,255)}

    number_dis = 0
    list_dis = []
    while True:
        vs = cap.read()[1]
        num_frame += 1
        frame = cv2.resize(vs,(640, 480))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        start = cv2.getTickCount()

        if select_method== 0:
            face = hog_dlib(frame, model)
            text_method = "HOG Dlib"
            color = color_value["red"]
        elif select_method == 1:
            face = haar(frame, model)
            text_method = "HAAR"
            color = color_value["green"]

        else :
            face = dnn(frame, model)
            text_method = "DNN"
            color = color_value["blue"]

        if face:
            number_dis += 1
            l = face[-1][0]
            t = face[-1][1]
            r = face[-1][2]
            b = face[-1][3]

            cv2.rectangle(frame,(l,t),(r,b),color_value["white"], 1)
            cv2.putText(frame,f"#{text_method}", (l,t-5), font, 0.5, color_value["white"], 1)

            shape = face_landmark(frame, l, t, r, b)

            for i in range(len(shape)):
                cv2.circle(frame, shape[i], 2, color, -1)

            extand_eye = lambda near, far: (far - near) // 2 + near

            r_e_x = extand_eye(shape[2][0], shape[3][0])
            r_e_y = extand_eye(shape[2][1], shape[3][1]) \
                if shape[3][1] > shape[2][1] else extand_eye(shape[3][1] , shape[2][1])

            l_e_x = extand_eye(shape[1][0], shape[0][0])
            l_e_y = extand_eye(shape[1][1], shape[0][1]) \
                if shape[0][1] > shape[1][1] else extand_eye(shape[0][1], shape[1][1])

            c_x = extand_eye(r_e_x, l_e_x)
            c_y = extand_eye(r_e_y, l_e_y)

            p = {"view_center": [c_x, c_y], "eye_right": [r_e_x, r_e_y], "eye_left": [l_e_x, l_e_y]}

            dis = distance(p["view_center"], p["eye_right"])["distance"]

            cv2.putText(frame, f"Distance: {dis} cm.", (20, 20), font, 0.5, color_value["white"], 1)

            list_dis.append(dis)

        cv2.imshow("output", frame)
        if cv2.waitKey(1) == 27 or number_dis == 500:
            break

    print(list_dis)

    np.savetxt(f"{text_method}_6_80_{'accuracy.csv'}", list_dis)

    print('AVG time: {:.2f}'.format(sum(list_time)/len(list_time)))

    cv2.destroyAllWindows()
    cap.release()