# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:33:11 2021

@author: hp
"""


# use the saved model
import joblib

import smile_test1

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import os
import sys
import random
# 存储位置
output_dir = 'D:/myworkspace/JupyterNotebook/Smile/person'
size = 64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img    

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/shape_predictor_68_face_landmarks.dat')

# OpenCv 调用摄像头
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 480)


def get_features(img_rd):

    # 输入:  img_rd:      图像文件
    # 输出:  positions_lip_arr:  feature point 49 to feature point 68, 20 feature points / 40D in all

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 计算68点坐标
    positions_68_arr = []
    faces = detector(img_gray, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        positions_68_arr.append(pos)

    positions_lip_arr = []
    # 将点 49-68 写入 CSV
    # 即 positions_68_arr[48]-positions_68_arr[67]
    for i in range(48, 68):
        positions_lip_arr.append(positions_68_arr[i][0])
        positions_lip_arr.append(positions_68_arr[i][1])

    return positions_lip_arr


while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数 faces
    faces = detector(img_gray, 0)
    # 检测到人脸
    if len(faces) != 0:
        # 提取单张40维度特征
        positions_lip_test = get_features(img_rd)

        # path of models
        path_models = "D:/myworkspace/JupyterNotebook/Smile/data/data_models/"

        # #########  LR  ###########
        LR = joblib.load(path_models+"model_LR.m")
        ss_LR = smile_test1.model_LR()
        X_test_LR = ss_LR.transform([positions_lip_test])
        y_predict_LR = str(LR.predict(X_test_LR)[0]).replace('0', "no smile").replace('1', "with smile")
        print("LR:", y_predict_LR)

        # #########  LSVC  ###########
        LSVC = joblib.load(path_models+"model_LSVC.m")
        ss_LSVC = smile_test1.model_LSVC()
        X_test_LSVC = ss_LSVC.transform([positions_lip_test])
        y_predict_LSVC = str(LSVC.predict(X_test_LSVC)[0]).replace('0', "no smile").replace('1', "with smile")
        print("LSVC:", y_predict_LSVC)

        # #########  MLPC  ###########
        MLPC = joblib.load(path_models+"model_MLPC.m")
        ss_MLPC = smile_test1.model_MLPC()
        X_test_MLPC = ss_MLPC.transform([positions_lip_test])
        y_predict_MLPC = str(MLPC.predict(X_test_MLPC)[0]).replace('0', "no smile").replace('1', "with smile")
        print("MLPC:", y_predict_MLPC)

        # #########  SGDC  ###########
        SGDC = joblib.load(path_models+"model_SGDC.m")
        ss_SGDC = smile_test1.model_SGDC()
        X_test_SGDC = ss_SGDC.transform([positions_lip_test])
        y_predict_SGDC = str(SGDC.predict(X_test_SGDC)[0]).replace('0', "no smile").replace('1', "with smile")
        print("SGDC:", y_predict_SGDC)

        print('\n')

        # 按下 'q' 键退出
        if kk == ord('q'):
            break
        if kk == ord('s'):
            index = 1
            while True:
                if (index <= 10):#存储10张人脸特征图像
                    print('Being processed picture %s' % index)
                    # 从摄像头读取照片
                    success, img = cap.read()
                    # 转为灰度图片
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 使用detector进行人脸检测
                    dets = detector(gray_img, 1)
 
                    for i, d in enumerate(dets):
                        x1 = d.top() if d.top() > 0 else 0
                        y1 = d.bottom() if d.bottom() > 0 else 0
                        x2 = d.left() if d.left() > 0 else 0
                        y2 = d.right() if d.right() > 0 else 0
 
                        face = img[x1:y1,x2:y2]
                        # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
 
                        face = cv2.resize(face, (size,size))
 
                        cv2.imshow('image', face)
 
                        cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
 
                        index += 1
                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        break
                else:
                    print('Finished!')
                    # 释放摄像头 release camera
                    cap.release()
                    # 删除建立的窗口 delete all the windows
                    cv2.destroyAllWindows()
                    break

    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
