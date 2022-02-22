# Importing the libraries
import tensorflow as tf
from tensorflow.keras.layer import Conv2d
from tensorflow.keras.layer import MaxPool2d
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# Part 1
# Preprocessing => prevent overfitting/ overtraining
# Apply transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    sheer_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=64,
    batch_size=32,
    class_mode='binary'
)

# Part 2
# Building the CNN
cnn = tf.keras.models.Sequential()

# Step 1) Convolution
cnn.add(Conv2d(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))

# Stap 2) Pooling
cnn.add(MaxPool2d(pool_size=2, stride=2))

# Adding a second convolutional & pooling layer
cnn.add(Conv2d(filters=32, kernel_size=3, activation='relu')) # remove input shape, it's only needed with the first layer
cnn.add(MaxPool2d(pool_size=2, stride=2))

# Stap 3) Flattening 
# => result of all the previous convolutions and pooling
# => into a 1D vector which will become the input of a future fully connected neural network
cnn.add()