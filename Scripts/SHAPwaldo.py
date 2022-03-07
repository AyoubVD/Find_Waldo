# Importing the libraries
from __future__ import print_function
import shap
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
import os

''' os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# As an unsafe, unsupported, undocumented workaround you can set the environment variable 
os.environ['KMP_DUPLICATE_LIB_OK']='True'  '''
# Proprocessing the test and training set
train_datagen = ImageDataGenerator(rescale = 1./255, #pixelwaarden normaliseren
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32, #hoeveel foto's per keer langs neural netwerk laat passeren
                                                 class_mode = 'binary')
#x_train = training_set
#fit from generator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#x_test = test_set

#os.listdir('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing')
#print('---------------------------------------------------')
#print(training_set[1][1])
#print('---------------------------------------------------')
#print(len(training_set))
#print(type(training_set))

class_names = ['Waldo', 'Not Waldo']
#y_train = class_names
#y_test = y_train

# Initialising the CNN
cnn = tf.keras.models.Sequential() # layers gaan op elkaar volgen

# Add convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Add convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten()) # full connection maken (zorgt voor features)

# Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # sigmoid = input transformeren naar 0-1

# Compile CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 1)

# Making a prediction
test_image = image.load_img('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing/waldo/12_2_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  print("Tested negative for Waldo :'(")
else:
  print('Tested positive for Waldo! :D')

''' 

batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0][0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=[64,64,3]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
 '''
x_train, y_train = next(training_set)
x_test, y_test = next(test_set)
#model.fit(x = training_set, validation_data = test_set, epochs = 1)
score = cnn.evaluate(test_set, verbose=0) 
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# select a set of background examples to take an expectation over
background = x_train[np.random.choice(x_train.shape[0], 100, replace=True)]


# explain predictions of the model (named cnn in this case) on three images
e = shap.DeepExplainer(cnn, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
tf.executing_eagerly() #outputs following
# keras is no longer supported, please use tf.keras instead.
# Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode. See PR #1483 for discussion.
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
print('----------------------------------------------')
shap_values = e.shap_values(x_test[1:5]) # Gave 2 errors:
# 1) AttributeError: module 'tensorflow.python.eager.backprop' has no attribute '_record_gradient'
# File "C:\Users\ayoub\AppData\Local\Temp\tmpvn75rorw.py", line 16, in tf__grad_graph
# out = ag__.converted_call(ag__.ld(self).model, (ag__.ld(shap_rAnD),), None, fscope)

# 2) File "C:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo\Scripts\SHAPwaldo.py", line 153, in <module>
# shap_values = e.shap_values(x_test[1:5])