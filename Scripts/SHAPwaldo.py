import shap
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

# Importing the libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
''' valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    )
# Proprocessing the test and training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(
    directory='C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
valid_generator = valid_datagen.flow_from_directory(
    directory='C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
training_set = train_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    directory='C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
) 
test_set = test_datagen.flow_from_directory('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#os.listdir('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/testing')
#print('---------------------------------------------------')
#print(training_set[1][1])
#print('---------------------------------------------------')
#print(len(training_set))
#print(type(training_set))
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

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False) 
'''