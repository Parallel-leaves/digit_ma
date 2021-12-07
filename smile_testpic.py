# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:26:58 2021

@author: hp
"""


# use the saved model
import joblib

from smile_dlib_tezhengdian import get_features
import smile_test1

import cv2

# path of test img
path_test_img = "C:/Users/28205/Documents/Tencent Files/2820535964/FileRecv/test_nosmile.jpg"

# 提取单张40维度特征
positions_lip_test = get_features(path_test_img)

# path of models
path_models = "D:/myworkspace/JupyterNotebook/Smile/data/data_models/"

print("The result of"+path_test_img+":")
print('\n')

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

img_test = cv2.imread(path_test_img)

img_height = int(img_test.shape[0])
img_width = int(img_test.shape[1])

# show the results on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_test, "LR:    "+y_predict_LR,   (int(img_height/10), int(img_width/10)), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
cv2.putText(img_test, "LSVC:  "+y_predict_LSVC, (int(img_height/10), int(img_width/10*2)), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
cv2.putText(img_test, "MLPC:  "+y_predict_MLPC, (int(img_height/10), int(img_width/10)*3), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
cv2.putText(img_test, "SGDC:  "+y_predict_SGDC, (int(img_height/10), int(img_width/10)*4), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img_test)
cv2.waitKey(0)
