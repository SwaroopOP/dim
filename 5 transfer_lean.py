import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

#Download and extract dataset
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
filename= os.path.join(os.getcwd(),"cats_and_dogs_filtered.zip")
tf.keras.utils.get_file(filename,url)
with zipfile.ZipFile("cats_and_dogs_filtered.zip","r") as zip_ref:
  zip_ref.extractall()

#Define data generators
train_dir=os.path.join(os.getcwd(),"cats_and_dogs_filtered","train")
validation_dir= os.path.join(os.getcwd(),"cats_and_dogs_filtered","validation")

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode="binary")
validation_generator=validation_datagen.flow_from_directory(validation_dir,
                                                            target_size=(150,150),
                                                            class_mode="binary")

#Load pre-trained VGG16 model
conv_base = VGG16(weights="imagenet",
                   include_top=False,
                   input_shape=(150,150,3))
#Freeze convolutionalbase layers
conv_base.trainable=False

#build model on top of the convolutional base
model= tf.keras.models.Sequential([
    conv_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
#compile the model
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
              metrics=["accuracy"])

#train model
history=model.fit(train_generator,
                  steps_per_epoch=100,
                  epochs=30,
                  validation_data=validation_generator,
                  validation_steps=50)


#show input and its predicted class
x,y_true=next(validation_generator)
y_pred=model.predict(x)
class_names=['cat','dog']
for i in range (len(x)):
  plt.imshow(x[i])
  plt.title(f'Predicted class: {class_names[int(round(y_pred[i][0]))]}, True class:{class_names[int(y_true[i])]}')
  plt.show()

#plot accuracy and loss over time
acc= history.history["accuracy"]
val_acc=history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs = range(1, len(acc)+1)
plt.plot(epochs,acc,"bo",label="Training acc")
plt.plot(epochs,val_acc,"b",label="validation acc")
plt.title("Training and validation accuracy")
plt.legend()
