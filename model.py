import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your dataset and create labels for classes
dataset_dir = 'database/training'
validation_dir='database/validation'
class_labels = ['kalai','prashanth','sairam']

# Parameters
input_shape = (224, 224, 3)
num_classes = len(class_labels)
batch_size = 20
epochs = 2  # Adjust as needed
learning_rate = 0.001  # Adjust as needed

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    
    
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

# Freeze layers of the VGG16 base model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom top model for classification
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Combine the base model and the custom top model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
)

# Save the trained model
model.save('face_recognition_model.h5')

