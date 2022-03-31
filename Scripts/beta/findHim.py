'''
from PIL import Image

im = Image.open('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/waldo/1_4_6.jpg')
im.show()
'''
# C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/waldo/
# C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/notwaldo/
# C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training/
# C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/
''' 
# I use this to check wether or not the path is existant and to see if there is a pic
from PIL import Image
# 
im = Image.open('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training/2_4_3.jpg')  
im.show()
bm = Image.open('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/1_0_0.jpg')  
bm.show() '''
 

# Importing the libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from PIL import Image
import os

def default_train_datagen():
    datagen = ImageDataGenerator( fill_mode='constant', dtype=int)
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    datagen.fit(train_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary'))
    return datagen
def default_test_datagen():
    #datagen = ImageDataGenerator( fill_mode='constant', dtype=int)
    test_datagen = ImageDataGenerator(rescale = 1./255,fill_mode='constant', dtype=int)
    test_set = test_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
    test_set.fit(test_set)
    return test_set

def findW():
    # Proprocessing the test and training set
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    training_set = train_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
    #os.listdir('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing')
    print(len(test_set))

    # Initialising the CNN
    cnn = tf.keras.models.Sequential()

    # Add convolution layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Add convolution layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

    # Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    # Flattening
    cnn.add(tf.keras.layers.Flatten())
    
    # Dropout -> increase acc
    cnn.add(tf.keras.layers.Dropout(0.2))

    # Full connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Output layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compile CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    cnn.fit(x = training_set, validation_data = test_set, epochs = 1)
    cnn.save('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Models/model1')
'''     #------------------
    test_image = image.load_img(img, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        img = Image.open(x).convert('L')
        img.save('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/results/'+x.split('/')[len(x.split('/'))-1])
        return('Imposter')
    else:
        return("That's him officer")
    #------------------ '''

def fitW(img):
    model = tf.keras.models.load_model('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Models/converted/keras_model.h5')
    # Making a prediction
    test_image = image.load_img(img, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        return("That's him officer")
    else:
        img = Image.open(img).convert('L')
        print(img)
        img.save('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/results/')
        return('Imposter')

