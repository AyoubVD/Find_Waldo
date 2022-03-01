'''
from PIL import Image

# creating a object
im = Image.open('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/waldo/1_4_6.jpg')
  
im.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = NullFormatter(5)
img = mpimg.imread('/images/64/waldo/1_4-6.jpg')
imgplot = plt.imshow(img)
plt.show() 
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
import os

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
class_names = ['Waldo', 'Not Waldo']

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

# Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 10)

# Making a prediction
test_image = image.load_img('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/waldo/12_2_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Not Waldo'
else:
  prediction = 'Waldo'

print(prediction)
''' config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.__version__)
file_path_train = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/mixedwaldo/'
file_path_test = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/pics/'
os.listdir(file_path_test)
# Part 1
# Preprocessing => prevent overfitting/ overtraining
# Apply transformations
train_datagen = ImageDataGenerator(
    file_path_train,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    file_path_train,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    file_path_test,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
) '''

""" 
# Part 2
# Building the CNN
cnn = tf.keras.models.Sequential()

# Step 1) Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))

# Stap 2) Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional & pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # remove input shape, it's only needed with the first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Stap 3) Flattening 
# => result of all the previous convolutions and pooling
# => into a 1D vector which will become the input of a future fully connected neural network
cnn.add(tf.keras.layers.Flatten())

# Stap 4) Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # units = hidden neurons

# Stap 5) Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part 3) Training the CNN
# Compiliing the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the training set and evaluate it on the test set
cnn.fit(x = training_set, validation_data=test_set, epochs=25)

# Part 4) Making a single prediction
file_path_waldo = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/waldo/1_4_6.jpg'
file_path_notwaldo = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/64/notwaldo/1_0_0.jpg'

test_image = image.load_img(file_path_dog, target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

print(training_set.class_indices)
result = cnn.predict(test_image)

if(result[0][0] == 1):
    prediction = 'waldo'
else:
    prediction = 'notwaldo'
    
print(prediction) """