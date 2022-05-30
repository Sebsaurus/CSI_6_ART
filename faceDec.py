from email import generator
from pickletools import optimize
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import RMSprop
from keras.preprocessing.image import train
import numpy as plt


model = tf.keras.models.Sequential()
                                                                                                                    
                                                                                                                    
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
                                                                                                                   
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
                                                                                                                   
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
                                                                                                                   
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
                                                                                                                 
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
                                                                                                                   
tf.keras.layers.Flatten(),
                                                                                                               
tf.keras.layers.Dense(512, activation='relu'),
                                                                                               
tf.keras.layers.Dense(1, activation='sigmoid')

model.compile(loss='binary_crossentropy',
optimizer=RMSprop(lr=0.001),
metrics='accuracy')

train_generator = tf.data.Dataset.from_generator(generator = lambda:generator('train'),
                                               output_types = (tf.float32, tf.float32),
                                               output_shapes = ((10,4), (10,1)))

validation_generator = tf.data.Dataset.from_generator(generator = lambda:generator('validation'),
                                             output_types = (tf.float32, tf.float32),
                                             output_shapes = ((10,4), (10,1)))

history = model.fit(train_generator,
steps_per_epoch=8,
epochs=15,
verbose=1,
validation_data = validation_generator,
validation_steps=8)

model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
verbose=1)

plt.figure()
lw = 2
plt.plot(color='Red',
lw=lw, label='ROC curve (area = %0.2f)')
plt.plot([0, 1], [0, 1], color='Rose', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()