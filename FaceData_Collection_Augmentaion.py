## Importing Packages 

import os 
import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
import random

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


# Taking image from webcam
def makefolder():
    name = input(print('Enter Your name'))
    folder_path = 'D:/anaconda3\My_Projects/my_work_proj/Face Recognition/Face_image_data/'+ name
    
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)
        print('File Created Sucessfully')
        
    else:
        print('Name is alreday taken, please try adding alphanumeric characters to your name')
        
        user_inp = int(input(print('You still wanna continue? 1 - yes, 0 - No')))
        
        if user_inp == 1: 
            return makefolder()
        
        else:
            return None,0
    
    return name,folder_path

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img = img + noise
    np.clip(img, 0., 255.)
    return img



def image_capture_n_aug():
    vid = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0 
    name,folder_path = makefolder()
    print(name)
    print(folder_path)
    if name != None:

        while True:
            ret,frame=vid.read()
            faces=face_cascade.detectMultiScale(frame,1.3, 5)

            for x,y,w,h in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                count=count+1

                img_path = folder_path+'/'+ str(count)+".jpg"
                cv2.imwrite(img_path, frame[y:y+h,x:x+w])

                cv2.imshow("WindowFrame", frame)
            cv2.waitKey(100)

            if count>100:
                break

        vid.release()
        cv2.destroyAllWindows() 

#         Working Data Augmentation part 
    
    folder_path = folder_path + '/'

    img_list = []
    for i in os.listdir(folder_path):

        img = plt.imread(folder_path+i)
        img = Image.fromarray(img, 'RGB').resize((112,112))
        img_list.append(np.array(img))

    img = np.array(img_list)


    datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    preprocessing_function=add_noise,
    fill_mode='constant')
    
    new_folder = 'D:/anaconda3\\My_Projects/my_work_proj/Face Recognition/Face_image_data/' + name +'_augmented'
    os.makedirs(new_folder)
    
    i = 0
    for batch in datagen.flow(img, batch_size=32,
                            save_to_dir= new_folder ,
                            save_prefix=name,
                            save_format='jpg'):    
        i += 1    
        if i > 2000:        
            break
    
    print('Image Augmentation is done sucessfully')

    return None