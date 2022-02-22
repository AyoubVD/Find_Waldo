# Importing the libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.__version__)
file_path_train = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/CNN_tut/Convolutional Operation/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set'
file_path_test = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/CNN_tut/Convolutional Operation/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set'

# Part 1
# Preprocessing => prevent overfitting/ overtraining
# Apply transformations
train_datagen = ImageDataGenerator(
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
    target_size=64,
    batch_size=32,
    class_mode='binary'
)


# Part 2
# Building the CNN
cnn = tf.keras.models.Sequential()

# Step 1) Convolution
cnn.add(tf.keras.layer.Conv2d(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))

# Stap 2) Pooling
cnn.add(tf.keras.layer.MaxPool2d(pool_size=2, stride=2))

# Adding a second convolutional & pooling layer
cnn.add(tf.keras.layer.Conv2d(filters=32, kernel_size=3, activation='relu')) # remove input shape, it's only needed with the first layer
cnn.add(tf.keras.layer.MaxPool2d(pool_size=2, stride=2))

# Stap 3) Flattening 
# => result of all the previous convolutions and pooling
# => into a 1D vector which will become the input of a future fully connected neural network
cnn.add(tf.keras.layer.Flatten())

# Stap 4) Full Connection
cnn.add(tf.keras.layer.Dense(units=128, activation='relu')) # units = hidden neurons

# Stap 5) Output Layer
cnn.add(tf.keras.layer.Dense(units=1, activation='sigmoid'))


# Part 3) Training the CNN
# Compiliing the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the training set and evaluate it on the test set
cnn.fit(x = training_set, validation_data=test_set, epochs=25)

# Part 4) Making a single prediction
file_path_dog = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/CNN_tut/Convolutional Operation/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_1.jpg'
file_path_cat = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/CNN_tut/Convolutional Operation/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_2.jpg'

test_image = image.load_img(file_path_dog, target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

print(training_set.class_indices)
result = cnn.predict(test_image)

if(result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)