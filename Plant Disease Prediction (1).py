

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
import glob
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D ,Flatten, BatchNormalization ,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt




IMAGE_SIZE= 128
BATCH_SIZE= 42
CHANNELS = 3

train_ds= tf.keras.preprocessing.image_dataset_from_directory(
    'train',
    shuffle=True,
    seed=42,
    color_mode='rgb',
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)



validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    'valid',
    image_size= (IMAGE_SIZE, IMAGE_SIZE),
    seed=42,
    shuffle=True,
    batch_size=BATCH_SIZE)


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
   'test',
    image_size= (IMAGE_SIZE, IMAGE_SIZE),
    seed=42,
    shuffle=True
)


urls= 'train'
class_names = os.listdir(urls)


for image_batch,label_batch in train_ds.take(1):
    img= image_batch[0].numpy().astype("uint8")
    plt.title(class_names[label_batch[0]])
    plt.imshow(img)


def filter_green(image):
    result = tf.image.rgb_to_hsv(image)
    return result


resize_and_rescale= tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal",input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
    layers.RandomZoom(0.2),
    layers.Rescaling(1./255),
    layers.RandomZoom(0.1),
])



n_classes=38

chanDim= -1
model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    tf.keras.layers.Conv2D(64 ,padding='same' ,kernel_size=3, activation='relu', input_shape=(128,128, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=3,strides=2),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"),
    tf.keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1568,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(38,activation="softmax")
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model = keras.Sequential()

model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(128,128,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))

model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1568,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(38,activation="softmax"))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.summary()


history = model.fit(train_ds,validation_data = validation_ds ,epochs = 2)

model.evaluate(test_ds)


for images_batch, labels_batch in test_ds.take(1):
    first_image= images_batch[0].numpy().astype('uint8')
    first_label= labels_batch[0].numpy()
    print("Image to predict")
    plt.imshow(first_image)
    print("actual label",class_names[first_label])
    
    prediction= model.predict(images_batch)
    print("predict",class_names[np.argmax(prediction[0])])


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

