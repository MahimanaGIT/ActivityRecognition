import numpy as np
import tensorflow as tf
from model import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.test.is_gpu_available())

EPOCH = 50
target_size = (100, 100)

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'training',  # This is the source directory for training images
        target_size=(target_size[0], target_size[1]),  # All images will be resized to 150x150
        batch_size=50,
        # shuffle=True,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

valid_generator = train_datagen.flow_from_directory(
        'validation',  # This is the source directory for training images
        target_size=target_size,  # All images will be resized to 150x150
        batch_size=50,
        # shuffle=True,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

model = Model(target_size).model

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy']
              )

# print(model.summary())              

history = model.fit(
      train_generator, validation_data = valid_generator,
    #   shuffle=True,
      steps_per_epoch=300,
      validation_steps=44,  
      epochs=EPOCH)

acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

import matplotlib.pyplot as plt

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)

plt.show()

# successive_outputs = [layer.output for layer in model.layers[1:]]

# visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
# img_path = '/media/mahimana/New Volume/ActivityRecognition/validation/Golf-Swing/002/7606-2_700812.jpg'
# # give the image array as input, let it be "x"
# img = load_img(img_path, target_size=(100, 100))  # this is a PIL image
# x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
# x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# # Rescale by 1/255
# x /= 255

# feature_maps = visualization_model.predict(x)
# layer_names = [layer.name for layer in model.layers[1:]]

# for layer_name, feature_map in zip(layer_names, feature_maps):
# 	if(len(feature_map.shape) == 4):
# 		# Visualizing output of CNN only
# 		n_features = feature_map.shape[-1] #Number of features in feature map
# 		size = feature_map.shape[1]

# 		display_grid = np.zeros((size, size*n_features))

# 		for i in range (n_features):
# 			# Postprocess the feature to make it visually palatable
# 			x = feature_map[0, :, :, i]
# 			x -= x.mean()
# 			x /= x.std()
# 			x *= 64
# 			x += 128
# 			x = np.clip(x, 0, 255).astype('uint8')
# 			# We'll tile each filter into this big horizontal grid
# 			display_grid[:, i * size : (i + 1) * size] = x
# 		# Display the grid
# 		scale = 20. / n_features
# 		plt.figure(figsize=(scale * n_features, scale))
# 		plt.title(layer_name)
# 		plt.grid(False)
# 		plt.imshow(display_grid, aspect='auto', cmap='viridis')
# plt.show()