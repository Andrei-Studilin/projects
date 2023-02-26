import os
import cv2

def get_etalon_frames(path, video):
    
    count = 0 #Счётчик кадров

    videoFile = video
    cap = cv2.VideoCapture(videoFile) #Загрузка видео

    while(cap.isOpened()):
    
        frameId = cap.get(1) #номер текущего кадра
        ret, frame = cap.read()
    
        if ret == True:
        
            filename = path + 'frame{0:04d}.jpg'.format(count)
            cv2.imwrite(filename, frame) #Сохраняем кадр
            count += 1
    
        else:
            break
        
    cap.release()
    
    #Создаём список изображений из эталонной директории
    etalon_frames = os.listdir(path)
    
    return etalon_frames
