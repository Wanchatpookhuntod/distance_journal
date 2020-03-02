import cv2
import dlib
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--face_roi', help='choose face roi, 0(hog) 1(haar) 2(dnn)', type=int, default=2)
args = parser.parse_args()

predict_landmark = dlib.shape_predictor("res/model/shape_predictor_5_face_landmarks.dat")

video = r"C:\Users\CCS win 10 2020\Google Drive\write_medium\res\video\JonSnow.mp4"

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
    select_method = args.face_roi
    model = ""

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

    cap = cv2.VideoCapture(video)

    num_frame = 0
    list_time = []
    text_method = ""
    color = ""

    font = cv2.FONT_HERSHEY_COMPLEX
    color_value = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "white": (255,255,255)}

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
            l = face[-1][0]
            t = face[-1][1]
            r = face[-1][2]
            b = face[-1][3]

            cv2.rectangle(frame,(l,t),(r,b),color_value["white"], 1)
            cv2.putText(frame,f"#{text_method}", (l,t-5), font, 0.5, color_value["white"], 1)

            landmark = face_landmark(frame, l, t, r, b)

            for i in range(len(landmark)):
                cv2.circle(frame, landmark[i], 2, color, -1)

        time = time_processing(start)
        list_time.append(time)
        time_show = f'{text_method} elapsed time: {time:.2f} ms/f'

        cv2.putText(frame, time_show, (20, 20), font, 0.5, color_value["white"], 1)

        cv2.imshow("output", frame)
        if cv2.waitKey(1) == 27 or num_frame == 500:
            break

    print('AVG time: {:.2f}'.format(sum(list_time)/len(list_time)))

    cv2.destroyAllWindows()
    cap.release()