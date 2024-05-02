import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Load the VGG16 model
model=VGG16()
#Load the image
img=load_img("/content/football.png",target_size=(224,224))
img_arr=img_to_array(img)
img_arr=np.expand_dims(img_arr,axis=0)
img_arr=preprocess_input(img_arr)

#predict the objects in the image
preds=model.predict(img_arr)
decoded_preds=decode_predictions(preds,top=3)[0]

#print the predicted objects and their application
for pred in decoded_preds:
  print(f"{pred[1]}: {pred[2]*100:.2f}%")



