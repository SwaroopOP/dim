import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

#Load the CIFAR-10 dataset
(x_train, y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

# normalise the pixel values to be between 0 and 1
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#convert the labbles to one-hot encoded vectors
y_train= keras.utils.to_categorical(y_train,num_classes=10)
y_test= keras.utils.to_categorical(y_test,num_classes=10)

#define the model architecture
model=keras.models.Sequential([
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train the model
history = model.fit(x_train,
                  y_train,
                  epochs=10,
                  batch_size=64,
                  validation_data=(x_test,y_test))

#save the model
model.save('cifar10_model.h5')

''' if error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context'''

#load the saved model
model=keras.models.load_model('cifar10_model.h5')

#load and preprocess the test image
img=Image.open('two.png')
img=img.resize((32,32))
img_array=np.array(img)

#convert the image to 3 channels(RGB) if it has 4 channels(RGBA)
if img_array.shape[-1]==4:
    img_array= img_array[..., :3]
    img_array = img_array.astype('float32')/255.0
    img_array = np.expand_dims(img_array,axis=0)

#make predictions on test image
predictions = model.predict(img_array)
#get predicted class label
class_label=np.argmax(predictions)

#print the predicted class labels
print('Predicted class label:',class_label)
