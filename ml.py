import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import random


train_images = []
test_images = []
train_labels = []
test_labels = []
class_reference = ["Ambulance","Firetruck","Traffic"]

#setting ambulance sound spectorgram for training
for i in range(1,171):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    train_images.append(img)
    train_labels.append(0)

#setting firetruck sound spectorgram for training
for i in range(201,371):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    train_images.append(img)
    train_labels.append(1)

#setting ambulance sound spectorgram for testing
for i in range(171,201):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    test_images.append(img)
    test_labels.append(0)

#setting firetruck sound spectorgram for testing
for i in range(371,401):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    test_images.append(img)
    test_labels.append(1)

#setting traffic sound spectorgram for training
for i in range(401,571):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    train_images.append(img)
    train_labels.append(2)

#setting traffic sound spectorgram for testing
for i in range(571,601):
    img=cv2.imread("sound_"+str(i)+".png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
    test_images.append(img)
    test_labels.append(2)

#converting all lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


#reducing the values to small ones
train_images = train_images / 255.0
test_images =  test_images  / 255.0

print(train_images.shape)


#Creating the model
#Input layer contains 28 x 28 = 784 nodes since images and 28 x 28 pixels
#Output softmax layer contains 3 nodes - Ambulance , Firetruck and Traffic 
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(84,84,1)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D((4,4)))
model.summary()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4*4*64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))


#Compiling the model
optimizer = keras.optimizers.RMSprop(learning_rate=0.00151)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



#Training the model for 500 epochs
ep=21
history = model.fit(train_images, train_labels, epochs=ep)

test_loss = history.history['loss']
iter=[]

for i in range(0,ep):
   iter.append(i)

plt.plot(iter,test_loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

#Loss and accuracy metrics for testing
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

img_1 = cv2.imread("sound_ambulance.png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
img_1 = img_1.astype('float32')
img_1 = np.expand_dims(img_1,0)

img_2 = cv2.imread("sound_firetruck.png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
img_2 = img_2.astype('float32')
img_2 = np.expand_dims(img_2,0)

img_3 = cv2.imread("sound_traffic.png",cv2.IMREAD_GRAYSCALE).reshape((84,84,1))
img_3 = img_3.astype('float32')
img_3 = np.expand_dims(img_3,0)


predictions = model.predict(img_1)
print(class_reference[np.argmax(predictions[0])])

predictions = model.predict(img_2)
print(class_reference[np.argmax(predictions[0])])

predictions = model.predict(img_3)
print(class_reference[np.argmax(predictions[0])])

#saving the model
keras_file = "MP_model.h5"
keras.models.save_model(model,keras_file)

#Converting keras model for Tensorflow Lite and saving it
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
mp_lite_model = converter.convert()
open("MP_model.tflite","wb").write(mp_lite_model)





