import tensorflow as tf
#Loading the data mnist dataset
(train_data,train_labels),(test_data,test_labels)=tf.keras.datasets.mnist.load_data()

#preprocess the data
train_data=train_data.reshape(60000,784)/255.0 
test_data=test_data.reshape(10000,784)/255.0 
train_labels=tf.keras.utils.to_categorical(train_labels) 
test_labels=tf.keras.utils.to_categorical(test_labels) 

# Model Architecture
model=tf.keras.models.Sequential([
    # sequential model
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])

#Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#Training the model
history=model.fit(train_data,train_labels,validation_data=(test_data,test_labels),epochs=10,batch_size=128)

import matplotlib.pyplot as plt
#Plot the training loss and validation loss
plt.figure(figsize=(12,5))
#Plot training loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Taining and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#Plot training accuracy and validation accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Taining and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
