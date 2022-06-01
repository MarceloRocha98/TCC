# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:23:33 2022

@author: marce
"""

import cv2
import mediapipe as mp
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ajustar qnd inicializa sem face e depois insere a face

# motorista olhando para tras ?


# ymax = y nariz + 0.3 altura - distancia vertical do nariz ate o olho
# xmin = xmin - 0.3 largura + distancia horizontal entre o olho e xmin
# xmax = xmax + 0.3 largura - distancia horizontal entre olho e xmax
# ajustar cte (0.3)
# aumentar os contornos da boundingbox (colocar stride) (dependendo do ajuste da cte n precisa)
"""
 (xmin, ymin)      (xmax, ymin)
|                |
|                |
|                |
(xmin, ymax)-----------(xmax, ymax)
"""
    
# For static images:
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.75) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
        
        # print(detection) # detection é uma classe
        bBox = detection.location_data.relative_bounding_box
        h, w, c = image.shape
        boundBox = int(bBox.xmin*w), int(bBox.ymin*h), int(bBox.width*w), int(bBox.height*h) # reescalando os valores da bounding box
        cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2 )
      

        right_ear_tragion = mp_face_detection.get_key_point(detection, 
                                                            mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
        
        left_ear_tragion = mp_face_detection.get_key_point(detection,
                                                           mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
         
        left_eye = mp_face_detection.get_key_point(
             detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
        
        right_eye = mp_face_detection.get_key_point(
             detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        
        nose_tip = mp_face_detection.get_key_point(detection,
                                                           mp_face_detection.FaceKeyPoint.NOSE_TIP)
        """
        retangulo gerado da seguinte forma:
            rectangle(image,(x1, y1), (x2,y2), (255,0,0))
        (x1,y1)=quina_sup_esquerda ----p1
        |
        |
        |------------------------------ (x2,y2)= lim_inferior
        
        obs: quina_sup_esquerda -> quina superior, na esquerda, da bounding box da face
             quina_inf_direita -> quina inferior, na direita, da bounding box da face
             
             
        Verificar a bounding box do mediapipe
        talvez:
        xmin,ymin -------
        |
        |
        |----------------
        
        """
        ctex = 0.72
        ctey = 0.2
        nose_tip = nose_tip.x*w, nose_tip.y*h
        left_eye = left_eye.x*w, left_eye.y*h
        right_eye = right_eye.x*w, right_eye.y*h
        # xmin = xmin - 0.3 largura + distancia horizontal entre o olho e xmin
        quina_sup_esquerda = boundBox[0] - int(ctex*boundBox[2]) + (int(left_eye[0])-boundBox[0]), boundBox[1] # xmin, ymin
        
        
        
        
        # quina_inf_direita = int(left_ear_tragion.x*w), int(left_ear_tragion.y*h)
        quina_inf_direita = boundBox[0]+boundBox[2], boundBox[1] + boundBox[3]

        # xmax = xmax + 0.3 largura - distancia horizontal entre olho e xmax
      
        x_lim_inf = quina_inf_direita[0] + ctex*boundBox[2] - ( (boundBox[0]+boundBox[2]) - right_eye[0])
        
        # ymax = y nariz + 0.3 altura - menor diferença vertical do nariz ate o olho
        
    
        y_lim_inf = int(nose_tip[1]) + ctey*boundBox[3] - ( int(nose_tip[1] - max(right_eye[1], left_eye[1])))
        
        p1 = quina_sup_esquerda[0] + boundBox[2], quina_sup_esquerda[1]
        # lim_inferior = int(nose_tip[0]+d), int(nose_tip[1])
        # lim_inferior = int(quina_inf_direita[0]), int(nose_tip[1])
        lim_inferior = int(x_lim_inf), int(y_lim_inf)
        
        
        res = cv2.rectangle(image, (quina_sup_esquerda[0], quina_sup_esquerda[1]), (lim_inferior[0], lim_inferior[1]), (255,0,0))
        # res = cv2.rectangle(image, (lim_inferior[0], lim_inferior[1]), (lim_inferior[0]+3, lim_inferior[1]+3), (255,0,0)) 
        # res = cv2.rectangle(image, (quina_sup_esquerda[0], quina_sup_esquerda[1]), (quina_sup_esquerda[0]+3, quina_sup_esquerda[1]+3), (255,0,0))  # verificacao do xmin,ymin no mediapipe
        roi = image[boundBox[1]:int(nose_tip[1]), boundBox[0]:quina_inf_direita[0]]
        
        # print(dir(results))
        # if(results is not None):
        if(len(res)):
            cv2.imwrite("my.png", res)
        if(len(roi)):
            cv2.imwrite("roi.png", roi)
        
        # cv2.rectangle(image, quina_sup_esquerda, quina_inf_direita, (255,0,0), 2)
        
        # left_eye = left_eye.x*w, left_eye.y*h # desnormalizacao
        # right_eye = right_eye.x*w, right_eye.y*h # desnormalizacao
        

        
        # print(mp_face_detection.get_key_point(
        #      detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
      
    end = time.time()
    totalTime = end - start
    fps = 1 /(totalTime+0.0000000001)
    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()