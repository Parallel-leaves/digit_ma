# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:23:42 2021

@author: hp
"""

import cv2
import numpy as np
import imutils
from pyzbar import pyzbar
def stackImages(scale, imgArray):
    """
        将多张图像压入同一个窗口显示
        :param scale:float类型，输出图像显示百分比，控制缩放比例，0.5=图像分辨率缩小一半
        :param imgArray:元组嵌套列表，需要排列的图像矩阵
        :return:输出图像
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


#读取图片
src = cv2.imread("C:\\Users\\28205\\Documents\\Tencent Files\\2820535964\\FileRecv\\txm.jpg")
img = src.copy()

#灰度
img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#高斯滤波
img_2 = cv2.GaussianBlur(img_1, (5, 5), 1)


#Sobel算子
sobel_x = cv2.Sobel(img_2, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_2, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
img_3 = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

#均值方波
img_4 = cv2.blur(img_3, (5, 5))

#二值化
ret, img_5 = cv2.threshold(img_4, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#闭运算
kernel = np.ones((100, 100), int)
img_6 = cv2.morphologyEx(img_5, cv2.MORPH_CLOSE, kernel)

#开运算
kernel = np.ones((200, 200), int)
img_7 = cv2.morphologyEx(img_6, cv2.MORPH_OPEN, kernel)

#绘制条形码区域
contours = cv2.findContours(img_7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
c = sorted(contours, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], -1, (0,255,0), 20)

#显示图片信息
cv2.putText(img, "results", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
cv2.putText(img_1, "gray", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
cv2.putText(img_2, "GaussianBlur",(200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
cv2.putText(img_3, "Sobel", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
cv2.putText(img_4, "blur", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
cv2.putText(img_5, "threshold", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
cv2.putText(img_6, "close", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
cv2.putText(img_7, "open", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

#输出条形码
barcodes = pyzbar.decode(src)
for barcode in barcodes:
    barcodeData = barcode.data.decode("utf-8")
    cv2.putText(img, barcodeData, (200, 600), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 30)

#显示所有图片
imgStack = stackImages(0.8, ([img_1, img_2],[img_3,img_4],[img_5,img_6],[img_7,img]))
cv2.imshow("imgStack", imgStack)
cv2.waitKey(0)

