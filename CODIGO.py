# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:47:37 2021

@author: Claudia
"""
# IMPORTANTE: ES NECESARIO TENER UN VIDEO (face-object.mp4) PARA QUE EL CÓDIGO 
# SE EJECUTE DE MANERA CORRECTA 

import cv2
import argparse

# Marcos para la cara
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])  
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,76,255), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

# Edades y generos
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(1-3)', '(5-7)', '(9-13)', '(15-20)', '(23-33)', '(35-47)', '(48-59)', '(65-100)']
genderList=['Male','Female']

# Valores netos de cara, edad y genero
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# Deteccion de objetos 

cap = cv2.VideoCapture(args.image if args.image else 'face-object.mp4')

padding = 20

cap.set(3, 640)
cap.set(4, 480)

classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# -------------------

while cv2.waitKey(1)<0:
    
    hasFrame,frame=cap.read()
    cv2.imshow('frame',frame)
    if not hasFrame:
        cv2.waitKey()
        break
    
    img,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')            

    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = 0.5)
    print(classIds, bbox)
    

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color = (0, 255, 0), thickness = 2)
            
            cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)     
            
            cv2.putText(img, f'{gender}, {age}', (faceBox[0], faceBox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (205,81,50), 2, cv2.LINE_AA)  
    

    cv2.imshow("Output", img)
    cv2.waitKey(1)
    
