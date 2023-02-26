#Загрузка библиотек
import torch
import torchvision
from torchvision import transforms as T
import cv2
import os
import glob
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.io import read_image
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

#Звгрузка скриптов
from skeleton import draw_skeleton_per_person
from affine import affine
from cosine_distance import cosine_distance
from weight_distance import weight_distance
from get_directory import get_directory
from get_frames import get_etalon_frames


#Загружаем предобученную модель
model = keypointrcnn_resnet50_fpn(pretrained=True)
model = model.eval()

#Трансформация для изображений
tr_image = torchvision.transforms.Compose([T.ToTensor()])

#Директория для кадров из эталонного видео
path = '../../MyProject_Models/images/etalon/'
get_directory(path)

#Кадры из эталонного видео
etalon_frames = get_etalon_frames(
                path,
                '../../MyProject_Data/VIDEO_6.mp4')

#Директориz для кадров из рабочего видео
work_path = '../../MyProject_Models/images/work/'
get_directory(work_path)

result_list = [] #Для сохранения результатов

counter = 0 #Счётчик для кадров из рабочего видео
et_counter = 0 #Счётчик для кадров из эталонного видео

videoFile = "../../MyProject_Data/VIDEO_5.mp4"
cap = cv2.VideoCapture(videoFile) #Загрузка видео

while(cap.isOpened()):
    frameId = cap.get(1) #Номер текущего кадра
    ret, frame = cap.read()
    
    if ret == True:
        
        filename = work_path + 'frame{0:04d}.jpg'.format(counter)
        cv2.imwrite(filename, frame)
        
        #Парами загружаем кадры
        vid = cv2.imread(filename)
        vid = cv2.rotate(vid, cv2.ROTATE_90_CLOCKWISE)
        et_vid = cv2.imread(path + etalon_frames[et_counter])
        et_vid = cv2.rotate(et_vid, cv2.ROTATE_90_CLOCKWISE)
        
        
        #Переводим в тензор
        etalon_img = tr_image(et_vid).cuda()
        work_img = tr_image(vid).cuda()
        
        #Подаём в модель
        model.cuda()
        with torch.no_grad():
            etalon_output = model([etalon_img])
            work_output = model([work_img])
        
        conf = etalon_output[0]['keypoints_scores'][0].detach().cpu().numpy()
        
        
        #Соберём два набора ключевых точек
        etalon_list = []
        work_list = []
        
        for kp in range(len(etalon_output[0]['keypoints'][0])):
            etalon_list.append(
        etalon_output[0]['keypoints'][0][kp, :2].detach().cpu().numpy())
        etalon_matrix = np.array(etalon_list)
        
        for kp in range(len(work_output[0]['keypoints'][0])):
            work_list.append(
        work_output[0]['keypoints'][0][kp, :2].detach().cpu().numpy())
        work_matrix = np.array(work_list)
        
        
        #Афинное преобразование
        input_transform = affine(etalon_matrix, work_matrix)
        
        
        #Считаем косинусное сходство
        cd = cosine_distance(etalon_matrix, input_transform)
        #Результат добавляем к изображению
        vid = cv2.putText(
                          vid,
                          str(cd),
                          (20,20),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (0,0,255), 
                          2)
        
        #Наносим скелет на рабочее изображение
        draw_skeleton_per_person(
                         filename,
                         vid,
                         work_output[0]['keypoints'],
                         work_output[0]['keypoints_scores'],
                         work_output[0]['scores'],
                         keypoint_threshold=2,
                         conf_threshold=0.9)
        
        #Считаем взвешенное совпадение
        wd = weight_distance(etalon_matrix, input_transform, conf)
        #Сохраняем результаты вычислений
        result_list.append(wd)
        
        print(wd)
        
        counter += 1 #переходим к следующему кадру
        
        #При достижении конца эталонного видео, обнуляем счётчик, запуская видео с начала
        if et_counter == len(etalon_frames)-1:
            et_counter = 0
        #Или переходим к следующему кадру    
        else:
            et_counter += 1
            
        
        
    else:
        break
        
cap.release()
