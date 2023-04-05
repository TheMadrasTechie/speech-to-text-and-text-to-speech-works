from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
cap = cv.VideoCapture(0)
head_shoulder = {}
emotion_detector = FER(mtcnn=True)
def res(img):        
    global head_shoulder
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    head_shoulder["head"] = []
    head_shoulder["right"] = []
    head_shoulder["left"] = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        #print(results.pose_landmarks)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape            
            cx, cy = int(lm.x * w), int(lm.y*h)
            if id == 0:
                cv.circle(img,(cx, cy), 5, (0,255,0), -1)
                #tmp.append(["Head : "+str(cx)+" "+str(cy)])
                head_shoulder["head"]=[int(cx),int(cy)]

            if id == 12:
                cv.circle(img,(cx, cy), 5, (0,255,0), -1)
                #tmp.append("Right : "+str(cx)+" "+str(cy))
                head_shoulder["right"]=[int(cx),int(cy)]
            if id == 11:
                cv.circle(img,(cx, cy), 5, (255,0,0), -1)
                #tmp.append("Left : "+str(cx)+" "+str(cy))
                head_shoulder["left"]=[int(cx),int(cy)]

def detection(input_image):
    print(type(input_image))
    result =[]  
    #if (type(input_image) is numpy.ndarray):
    result = emotion_detector.detect_emotions(input_image)
    # else:
    #   pass
    if len(result) > 0:
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(input_image,(bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255), 2,)
        emotion_name, score = emotion_detector.top_emotion(input_image )
        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (211, 211,211) 
            if score < 0.01:
                pass
            else:       
                color =  (0, 155, 255)
            emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))
            cv2.putText(input_image,emotion_score,(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
    else:
        print("No Face")
    return result

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:        
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res(frame)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # print(center_left)
            # print(center_right)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)        
            
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()