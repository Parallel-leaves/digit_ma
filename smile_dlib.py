# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:07:26 2021

@author: hp
"""


import cv2                     #  图像处理的库 OpenCv
import dlib                    # 人脸识别的库 dlib
import numpy as np             # 数据处理的库 numpy
class face_emotion():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("D:\\shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        self.cnt = 0  
    def learning_face(self):
        line_brow_x = []
        line_brow_y = []
        while(self.cap.isOpened()):

            flag, im_rd = self.cap.read()
            k = cv2.waitKey(1)
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  
            faces = self.detector(img_gray, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
     
            # 如果检测到人脸
            if(len(faces) != 0):
                
                # 对每个人脸都标出68个特征点
                for i in range(len(faces)):
                    for k, d in enumerate(faces):
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255))
                        self.face_width = d.right() - d.left()
                        shape = self.predictor(im_rd, d)
                        mouth_width = (shape.part(54).x - shape.part(48).x) / self.face_width 
                        mouth_height = (shape.part(66).y - shape.part(62).y) / self.face_width
                        brow_sum = 0 
                        frown_sum = 0 
                        for j in range(17, 21):
                            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                            frown_sum += shape.part(j + 5).x - shape.part(j).x
                            line_brow_x.append(shape.part(j).x)
                            line_brow_y.append(shape.part(j).y)

                        tempx = np.array(line_brow_x)
                        tempy = np.array(line_brow_y)
                        z1 = np.polyfit(tempx, tempy, 1)  
                        self.brow_k = -round(z1[0], 3) 
                        
                        brow_height = (brow_sum / 10) / self.face_width # 眉毛高度占比
                        brow_width = (frown_sum / 5) / self.face_width  # 眉毛距离占比

                        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y + 
                                   shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
                        eye_hight = (eye_sum / 4) / self.face_width
                        if round(mouth_height >= 0.03) and eye_hight<0.56:
                            cv2.putText(im_rd, "smile", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                            (0,255,0), 2, 4)

                        if round(mouth_height<0.03) and self.brow_k>-0.3:
                            cv2.putText(im_rd, "unsmile", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                        (0,255,0), 2, 4)
                cv2.putText(im_rd, "Face-" + str(len(faces)), (20,50), font, 0.6, (0,0,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(im_rd, "No Face", (20,50), font, 0.6, (0,0,255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "S: screenshot", (20,450), font, 0.6, (255,0,255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "Q: quit", (20,470), font, 0.6, (255,0,255), 1, cv2.LINE_AA)
            if (cv2.waitKey(1) & 0xFF) == ord('s'):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)
            # 按下 q 键退出
            if (cv2.waitKey(1)) == ord('q'):
                break
            # 窗口显示
            cv2.imshow("Face Recognition", im_rd)
        self.cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()