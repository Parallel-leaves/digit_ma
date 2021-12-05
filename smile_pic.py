# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:20:17 2021

@author: hp
"""

# 单张图片进行判断  是笑脸还是非笑脸
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
#加载模型
model = load_model('smileAndUnsmile1.h5')
#本地图片路径
img_path='C:\\Users\\28205\\Documents\\Tencent Files\\2820535964\\FileRecv\\test_nosmile.jpg'
img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)/255.0
img_tensor = np.expand_dims(img_tensor, axis=0)
prediction =model.predict(img_tensor)  
print(prediction)
if prediction[0][0]>0.5:
    result='非笑脸'
else:
    result='笑脸'
print(result)
