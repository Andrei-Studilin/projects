#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob

def get_directory(path):
    #Создаём директорию, если она не существует
    if not os.path.exists(path):
        os.mkdir(path)
    #Или очищаем существующую
    else:
        files = glob.glob(path + '*.jpg')
        for file in files:
            os.remove(file)

