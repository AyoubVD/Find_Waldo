import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf
import imgaug.augmenters as iaa
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

data_dir = pathlib.Path('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/Images/training/')
#print(data_dir)

#for x in os.listdir('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/Images/training/waldo clean'):
#        print(x)

waldo_clean = list(data_dir.glob('waldo clean/*'))
waldo = []

for x in waldo_clean:
        waldo.append(str(x))
#image = np.array(waldo_clean[0])
image = cv2.imread(waldo[0])

#cv2.imshow("image",image)
#cv2.waitKey(0)

images = []
for img_path in waldo:
    img = cv2.imread(img_path)
    images.append(img)
    
# 2. Image Augmentation
augmentation = iaa.Sequential([
    # 1. Flip
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),

    # 2. Affine
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
               rotate=(-30, 30),
               scale=(0.5, 1.5)),

    # 3. Multiply
    iaa.Multiply((0.8, 1.2)),


])
directo = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/Images/training/waldo clean/'
x = 0
# 3. Show Images
while x<60:
    augmented_images = augmentation(images=images)
    for img in augmented_images:
        
        cv2.imwrite(directo +str(x)+".png", img)
        x+=1
