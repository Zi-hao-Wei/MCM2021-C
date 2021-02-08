import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image
 
 
def Video2Pic(videoPath,imgPath):
    # videoPath = "youvideoPath"  # 读取视频路径
    # imgPath = "youimgPath"  # 保存图片路径
    
    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        cv2.imshow('frame',frame)
        p=imgPath + str(frame_count).zfill(4)+".jpg"
        print(p)
        cv2.imwrite(p, frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")
 

if __name__ == '__main__':
    videoPath = "ATT2800_trim.55AB752B-AE68-452A-A7B5-99B1B090397A.MOV"  # 读取视频路径
    imgPath = "E:\\MCM2021\\Test\\"  # 保存图片路径
    Video2Pic(videoPath,imgPath)