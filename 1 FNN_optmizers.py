import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#Load dataset
iris=load_iris()
#get features and input
X=iris.data
y= iris.target
#One-hot encode labels
lb=LabelBinarizer()
y=lb.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)

#Model Architecture
model=tf.keras.Sequential([
    #three layers
    #Output Layer is softmax
    #1&2 layers use relu activation function
    tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile Model with different optimizers
optimizers=['sgd','adam','rmsprop']
for optimizer in optimizers:
  model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  #Training the model
  history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,verbose=0)
  #Evaluate model
  loss,accuracy =model.evaluate(X_test,y_test,verbose=0)
  print('Optimizer:',optimizer)
  print('Test Loss',loss)
  print('Test accuracy',accuracy)

#Allow user to input the values for the flower attributes
print('\nInput values for the flower attributes')
sepal_length=float(input('Sepal Length (cm):'))
sepal_width=float(input('Sepal Width (cm):'))
petal_length=float(input('Petal Length (cm):'))
petal_width=float(input('Petal Width (cm):'))

#Predict class of flower based on input values
input_values=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
prediction =model.predict(input_values)
predicted_class=np.argmax(prediction)
class_names=iris.target_names
print('\nPredicted class:',class_names[predicted_class])
